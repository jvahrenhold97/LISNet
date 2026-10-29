import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from model import serialization
from utilities.logging import prog_state
import scipy.ndimage as ndi


class GlobalConsensusInference:
    """Global consensus inference engine for high-resolution instance segmentation."""
    def __init__(self, full_H: int, full_W: int, mask_thresh: float = 0.45) -> None:
        self.full_H = full_H
        self.full_W = full_W
        self.mask_thresh = mask_thresh
        self.master_sem_logits = np.zeros((3, full_H, full_W), dtype=np.float32)
        self.master_sem_count = np.zeros((full_H, full_W), dtype=np.float32)
        self.candidates = []

    def add_candidates(self, gy: int, gx: int, masks_res: torch.Tensor, scores_res: torch.Tensor, win_weight: np.ndarray, sem_res: torch.Tensor, flip_code: int | None = None) -> None:
        if flip_code is not None:
            dims = [-1] if flip_code == 1 else ([-2] if flip_code == 0 else [-2, -1])
            sem_res = torch.flip(sem_res, dims)
        sem_np = sem_res[0].cpu().numpy()
        h_s, w_s = sem_np.shape[1:]
        self.master_sem_logits[:, gy:gy+h_s, gx:gx+w_s] += sem_np
        self.master_sem_count[gy:gy+h_s, gx:gx+w_s] += 1.0
        valid_idx = scores_res >= 0.25
        if not valid_idx.any():
            return
        v_masks = masks_res[valid_idx]
        v_scores = scores_res[valid_idx]
        if flip_code is not None:
            v_masks = torch.flip(v_masks, dims)
        m_probs = torch.sigmoid(v_masks)
        mask_bins = m_probs > self.mask_thresh
        has_pixels = mask_bins.view(mask_bins.shape[0], -1).any(dim=1)
        if not has_pixels.any():
            return
        m_probs = m_probs[has_pixels].cpu().numpy()
        mask_bins = mask_bins[has_pixels].cpu().numpy()
        v_scores = v_scores[has_pixels].cpu().numpy()
        win_h, win_w = win_weight.shape
        for i in range(len(v_scores)):
            score = v_scores[i]
            mask_bin = mask_bins[i]
            x, y, w, h = cv2.boundingRect(mask_bin.astype(np.uint8))
            y0, y1 = y, y + h
            x0, x1 = x, x + w
            mask_crop = mask_bin[y0:y1, x0:x1]
            dist_crop = cv2.distanceTransform(
                mask_crop.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            dist_max = dist_crop.max()
            if dist_max > 0:
                dist_crop /= dist_max
            priority_map_crop = score * win_weight[y0:y1, x0:x1] * dist_crop
            avg_priority = priority_map_crop[mask_crop].mean()
            t_win, l_win = y0 <= 1, x0 <= 1
            b_win, r_win = y1 >= win_h-2, x1 >= win_w-2
            is_internal = not (
                (t_win and gy > 0) or 
                (l_win and gx > 0) or 
                (b_win and (gy + win_h) < self.full_H) or 
                (r_win and (gx + win_w) < self.full_W)
            )
            self.candidates.append({
                'pos': (gy + y0, gx + x0),
                'mask': mask_crop,
                'priority_map': priority_map_crop,
                'avg_priority': avg_priority,
                'is_internal': is_internal
            })

    def resolve(self) -> np.ndarray:
        self.master_sem_logits /= np.maximum(self.master_sem_count, 1.0)
        sem_map = np.argmax(self.master_sem_logits, axis=0)
        self.candidates.sort(key=lambda x: x['avg_priority'], reverse=True)
        inst_map = np.zeros((self.full_H, self.full_W), dtype=np.int32)
        score_map = np.zeros((self.full_H, self.full_W), dtype=np.float32)
        next_id = 1
        for cand in self.candidates:
            gy, gx = cand['pos']
            mh, mw = cand['mask'].shape
            roi_inst = inst_map[gy:gy+mh, gx:gx+mw]
            roi_score = score_map[gy:gy+mh, gx:gx+mw]
            M = cv2.moments(cand['mask'].astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cy, cx = mh // 2, mw // 2
            anchor_id = roi_inst[cy, cx]
            if anchor_id > 0:
                update = cand['mask'] & (cand['priority_map'] > (roi_score - 0.05))
                update &= ((roi_inst == 0) | (roi_inst == anchor_id))
                roi_inst[update] = anchor_id
                roi_score[update] = cand['priority_map'][update]
            else:
                if cand['is_internal'] or cand['avg_priority'] > 0.45:
                    empty_mask = cand['mask'] & (roi_inst == 0)
                    empty_ratio = empty_mask.sum() / (cand['mask'].sum() + 1e-6)
                    if empty_ratio > 0.65:
                        roi_inst[empty_mask] = next_id
                        roi_score[empty_mask] = cand['priority_map'][empty_mask]
                        next_id += 1
        inst_map[np.isin(sem_map, [0, 255])] = 0
        return self._finalize(inst_map)

    def _finalize(self, inst_map: np.ndarray) -> np.ndarray:
        final = np.zeros_like(inst_map)
        slices = ndi.find_objects(inst_map)
        curr_id = 1
        for uid_minus_1, slc in enumerate(slices):
            if slc is None:
                continue
            uid = uid_minus_1 + 1
            mask_crop = inst_map[slc] == uid
            filled_crop = ndi.binary_fill_holes(mask_crop)
            if filled_crop.sum() > 40:
                final[slc][filled_crop] = curr_id
                curr_id += 1
        return final


def get_global_coords(y: int, x: int, H_pad: int, W_pad: int, flip_code: int | None, crop_size: int) -> tuple[int, int]:
    """Transform local sliding window coordinates back to global image coordinates."""
    if flip_code is None: return y, x
    if flip_code == 1:    return y, W_pad - x - crop_size
    if flip_code == 0:    return H_pad - y - crop_size, x
    if flip_code == -1:   return H_pad - y - crop_size, W_pad - x - crop_size
    return y, x


def color_and_save_visualization(image_rgb: np.ndarray, inst_map: np.ndarray, save_path: str) -> None:
    """Create colored instance visualization with boundary highlights and save."""
    vis_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    unique_ids = np.unique(inst_map.astype(np.int32))
    unique_ids = unique_ids[unique_ids > 0]
    assigned_colors = {}
    kernel = np.ones((3, 3), np.uint8)
    for inst_id in unique_ids:
        mask = (inst_map == inst_id)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        neighbor_ids = np.unique(inst_map[dilated > 0])
        neighbor_ids = neighbor_ids[(neighbor_ids > 0) & (neighbor_ids != inst_id)]
        neighbor_colors = [assigned_colors[nid] for nid in neighbor_ids if nid in assigned_colors]
        while True:
            new_color = np.random.randint(30, 255, (3,)).tolist()
            if any(all(abs(new_color[i] - c[i]) < 45 for i in range(3)) for c in neighbor_colors):
                continue
            break
        assigned_colors[inst_id] = new_color
        vis_image[mask] = new_color
    cv2.imwrite(save_path, vis_image)


def run(device: torch.device, config: dict) -> None:
    """Run inference on test images if available with both images and masks, otherwise on root images."""
    use_sw = config.get("use_sliding_window", True)
    use_ms = config.get("use_multiscale", False)
    use_tta = config.get("use_tta", True)
    window_size = int(config.get("inference_window_size"))
    stride = int(config.get("inference_stride", window_size // 4))
    ROOT = config.get("root")
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    test_dir = os.path.join(ROOT, "test")
    test_img_dir = os.path.join(test_dir, "images")
    test_mask_dir = os.path.join(test_dir, "masks")
    if not os.path.isdir(test_mask_dir):
        test_mask_dir = os.path.join(test_dir, "leaf_instances")

    has_test_images = os.path.isdir(test_img_dir) and any(f.lower().endswith(valid_exts) for f in os.listdir(test_img_dir))
    has_test_masks = os.path.isdir(test_mask_dir) and any(f.lower().endswith(valid_exts) for f in os.listdir(test_mask_dir))

    if has_test_images and has_test_masks:
        IMAGES_ROOT = test_img_dir
        OUTPUT = os.path.join(test_dir, "inference")
        prog_state(f"[*] Found labeled test set at: {test_dir}")
    else:
        IMAGES_ROOT = ROOT
        OUTPUT = os.path.join(ROOT, "inference")
        prog_state(f"[*] No complete test set found. Running inference directly on root: {ROOT}")

    os.makedirs(os.path.join(OUTPUT, "colored"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "leaf_instances"), exist_ok=True)

    model = serialization.load_for_inference(config.get("weights"), device)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image_list = sorted([f for f in os.listdir(IMAGES_ROOT) if f.lower().endswith(valid_exts)])
    for img_idx, single_image in enumerate(image_list):
        prog_state(f"Processing Image {img_idx+1}/{len(image_list)}: {single_image}")
        img_raw = cv2.imread(os.path.join(IMAGES_ROOT, single_image))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = img_rgb.shape[:2]
        pad_unit = window_size if use_sw else 32
        pad_h = (pad_unit - (H_orig % pad_unit)) % pad_unit
        pad_w = (pad_unit - (W_orig % pad_unit)) % pad_unit
        padded_img = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        H_pad, W_pad = padded_img.shape[:2]
        engine = GlobalConsensusInference(H_pad, W_pad)
        if not use_sw:
            scales = [("FULL", None)]
        else:
            if use_ms:
                scales = [
                    (max(64, window_size // 2), max(16, stride // 2)),
                    (window_size, stride)
                ]
            else:
                scales = [(window_size, stride)]
        tta_modes = [("Orig", None), ("FlipH", 1), ("FlipV", 0), ("FlipHV", -1)] if use_tta else [("Orig", None)]
        for tta_name, f_code in tta_modes:
            curr_img = cv2.flip(padded_img, f_code) if f_code is not None else padded_img
            for c_size, s_val in scales:
                if c_size == "FULL":
                    c_h, c_w, s_y, s_x = H_pad, W_pad, H_pad, W_pad
                    m_in_h, m_in_w = H_pad, W_pad
                else:
                    c_h, c_w, s_y, s_x = c_size, c_size, s_val, s_val
                    m_in_h, m_in_w = window_size, window_size
                coords = [(y, x) for y in range(0, H_pad - c_h + 1, s_y) 
                                 for x in range(0, W_pad - c_w + 1, s_x)]
                for y, x in tqdm(coords, desc=f"  └─ {tta_name} Scale {c_size}", leave=False):
                    gy, gx = get_global_coords(y, x, H_pad, W_pad, f_code, c_w)
                    h_y = np.hanning(c_h)
                    h_x = np.hanning(c_w)
                    if gy == 0: h_y[:c_h//2] = 1.0
                    if gy + c_h >= H_pad: h_y[c_h//2:] = 1.0
                    if gx == 0: h_x[:c_w//2] = 1.0
                    if gx + c_w >= W_pad: h_x[c_w//2:] = 1.0
                    h_win = np.outer(h_y, h_x)
                    h_win = (h_win * 0.9 + 0.1).astype(np.float32)
                    crop = curr_img[y:y+c_h, x:x+c_w]
                    crop_resized = cv2.resize(crop, (m_in_w, m_in_h)) if (c_h != m_in_h or c_w != m_in_w) else crop
                    input_tensor = transform(image=crop_resized)["image"].unsqueeze(0).to(device)
                    with torch.no_grad():
                        fields, masks, scores, _ = model(input_tensor)
                    masks_res = F.interpolate(masks[0].unsqueeze(1), size=(c_h, c_w), mode='bilinear')[:, 0]
                    sem_res = F.interpolate(fields["semantic"], size=(c_h, c_w), mode='bilinear')
                    engine.add_candidates(gy, gx, masks_res, scores[0], h_win, sem_res, f_code)
        final_inst_padded = engine.resolve()
        final_inst = final_inst_padded[:H_orig, :W_orig]
        stem = os.path.splitext(single_image)[0]
        cv2.imwrite(os.path.join(OUTPUT, "leaf_instances", f"{stem}.png"), final_inst.astype(np.uint16))
        color_and_save_visualization(img_rgb, final_inst, os.path.join(OUTPUT, "colored", f"{stem}_vis.jpg"))
    prog_state("Inference Complete.")