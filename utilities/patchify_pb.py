import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


def get_patch_coordinates(img_h: int = 1024, img_w: int = 1024, win_size: int = 512, stride: int = 256) -> list[tuple[int, int, int, int]]:
    """Compute (y1, y2, x1, x2) bounding boxes for 9 regular patches across a 1024x1024 image."""
    crops = []
    y_coords = [0, stride, img_h - win_size]
    x_coords = [0, stride, img_w - win_size]
    for y in y_coords:
        for x in x_coords:
            crops.append((y, y + win_size, x, x + win_size))
    return crops


def process_and_patchify_dataset(source_root: str, target_root: str) -> None:
    """Convert and tile PhenoBench images and masks into 512x512 patches with continuous instance IDs."""
    splits = ["train", "val"]
    patch_boxes = get_patch_coordinates(img_h=1024, img_w=1024, win_size=512, stride=256)
    valid_exts = (".png", ".jpg", ".jpeg")

    for split in splits:
        src_split_dir = os.path.join(source_root, split)
        if not os.path.isdir(src_split_dir):
            continue

        src_img_dir = os.path.join(src_split_dir, "images")
        src_sem_dir = os.path.join(src_split_dir, "semantics")
        src_vis_dir = os.path.join(src_split_dir, "leaf_visibility")
        
        src_inst_dir = os.path.join(src_split_dir, "leaf_instances")
        if not os.path.isdir(src_inst_dir):
            src_inst_dir = os.path.join(src_split_dir, "masks")

        tgt_img_dir = os.path.join(target_root, split, "images")
        tgt_mask_dir = os.path.join(target_root, split, "masks")
        tgt_sem_dir = os.path.join(target_root, split, "semantics")
        tgt_vis_dir = os.path.join(target_root, split, "leaf_visibility")

        for d in [tgt_img_dir, tgt_mask_dir, tgt_sem_dir, tgt_vis_dir]:
            os.makedirs(d, exist_ok=True)

        image_files = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith(valid_exts)])
        print(f"\n[INFO] Processing split '{split}': {len(image_files)} source images found.")

        for fname in tqdm(image_files, desc=f"Converting & Patching {split}"):
            stem, ext = os.path.splitext(fname)

            img_p = os.path.join(src_img_dir, fname)
            sem_p = os.path.join(src_sem_dir, fname)
            inst_p = os.path.join(src_inst_dir, fname)
            vis_p = os.path.join(src_vis_dir, fname)

            image = cv2.imread(img_p)
            sem_raw = cv2.imread(sem_p, cv2.IMREAD_UNCHANGED)
            inst_raw = cv2.imread(inst_p, cv2.IMREAD_UNCHANGED)
            vis_raw = cv2.imread(vis_p, cv2.IMREAD_UNCHANGED)

            if image is None or sem_raw is None or inst_raw is None:
                continue

            h, w = image.shape[:2]
            sem_raw = sem_raw.astype(np.int32)
            inst_raw = inst_raw.astype(np.int32)

            is_leaf = (sem_raw == 1) & (inst_raw > 0)
            is_void = (sem_raw == 255) | (sem_raw == 3) | (sem_raw == 4)

            full_visibility = np.zeros((h, w), dtype=np.uint8)
            if vis_raw is not None:
                vis_norm = vis_raw if vis_raw.dtype == np.uint8 else (vis_raw * 255.0).astype(np.uint8)
                full_visibility[is_leaf] = vis_norm[is_leaf]
            else:
                full_visibility[is_leaf] = 255

            for p_idx, (y1, y2, x1, x2) in enumerate(patch_boxes):
                patch_suffix = f"_PATCH{p_idx:02d}.png"
                patch_name = f"{stem}{patch_suffix}"

                img_patch = image[y1:y2, x1:x2]
                inst_raw_patch = inst_raw[y1:y2, x1:x2]
                is_leaf_patch = is_leaf[y1:y2, x1:x2]
                is_void_patch = is_void[y1:y2, x1:x2]
                vis_patch = full_visibility[y1:y2, x1:x2]

                inst_patch = np.zeros((512, 512), dtype=np.uint8)
                inst_patch[is_void_patch] = 255

                patch_leaf_ids = np.unique(inst_raw_patch[is_leaf_patch])
                patch_leaf_ids = patch_leaf_ids[patch_leaf_ids > 0]

                for new_id, orig_id in enumerate(patch_leaf_ids, start=1):
                    assigned_id = min(new_id, 254)
                    leaf_mask = is_leaf_patch & (inst_raw_patch == orig_id)
                    inst_patch[leaf_mask] = assigned_id

                sem_patch = np.zeros((512, 512), dtype=np.uint8)
                sem_patch[(inst_patch >= 1) & (inst_patch <= 254)] = 1
                sem_patch[inst_patch == 255] = 255

                cv2.imwrite(os.path.join(tgt_img_dir, patch_name), img_patch)
                cv2.imwrite(os.path.join(tgt_mask_dir, patch_name), inst_patch)
                cv2.imwrite(os.path.join(tgt_sem_dir, patch_name), sem_patch)
                cv2.imwrite(os.path.join(tgt_vis_dir, patch_name), vis_patch)

    print(f"\n[DONE] Successfully created patchified dataset at: {target_root}")


if __name__ == "__main__":
    PHENOBENCH_SOURCE_DIR = "Path/To/PhenoBench"
    PATCHIFIED_TARGET_DIR = "Output/Path/For/Dataset"

    process_and_patchify_dataset(
        source_root=PHENOBENCH_SOURCE_DIR,
        target_root=PATCHIFIED_TARGET_DIR
    )