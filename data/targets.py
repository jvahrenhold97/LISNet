import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


def generate_center_targets(
        instance_mask,
        sigma_scale=0.2,
        sigma_min=0.5,
        sigma_max=4.0,
        max_sigma_ratio=1.0,
        neighbor_clip=1.0,
        target_stride_geo=4.0, 
        target_stride_sem=2.0, 
        amb_radius=1,          
    ):
    """Generate center heatmap and offset targets from instance segmentation mask.
    
    Creates Gaussian heatmaps centered at instance centroids with adaptive sigma based on
    instance size. Generates offset maps for coordinate refinement and validity masks.
    
    Args:
        instance_mask (np.ndarray): Instance segmentation map (H, W) with positive integer IDs.
        sigma_scale (float): Gaussian sigma scaling factor relative to instance radius. Defaults to 0.2.
        sigma_min (float): Minimum sigma value. Defaults to 0.5.
        sigma_max (float): Maximum sigma value. Defaults to 4.0.
        max_sigma_ratio (float): Max sigma relative to instance radius. Defaults to 1.0.
        neighbor_clip (float): Clipping factor for neighbor-aware sigma. Defaults to 1.0.
        target_stride_geo (float): Stride for geometry targets (centers, offsets). Defaults to 4.0.
        target_stride_sem (float): Stride for semantic targets. Defaults to 2.0.
        amb_radius (int): Radius for ambiguity detection at instance boundaries. Defaults to 1.
    
    Returns:
        dict: Target tensors with keys 'centers', 'center_dist', 'offsets', 'offset_mask',
              'valid_spatial_mask_s2', 'valid_spatial_mask_s4'.
    """
    H, W = instance_mask.shape
    inst_ds = instance_mask.astype(np.int64)
    offsets = np.zeros((2, H, W), dtype=np.float32)
    offset_mask = np.zeros((H, W), dtype=np.float32)
    ambiguous = np.zeros((H, W), dtype=bool)
    if amb_radius and amb_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * amb_radius + 1, 2 * amb_radius + 1))
        ids = np.unique(inst_ds)
        ids = ids[ids > 0]
        if len(ids) > 0:
            fg = inst_ds > 0
            for inst_id in ids:
                cur = inst_ds == inst_id
                if not cur.any(): continue
                other = fg & (~cur)
                if not other.any(): continue
                other_dil = cv2.dilate(other.astype(np.uint8), kernel).astype(bool)
                ambiguous |= cur & other_dil
    center_dist = np.ones((H, W), dtype=np.float32)
    erode_kernel = np.ones((3, 3), np.uint8)
    snapped_centers = []
    areas = []
    ids = np.unique(inst_ds)
    ids = ids[ids > 0]
    for inst_id in ids:
        ys_i, xs_i = np.where(inst_ds == inst_id)
        cx0, cy0 = xs_i.mean(), ys_i.mean()
        area0 = len(xs_i)
        cur = inst_ds == inst_id
        cur_er = cv2.erode(cur.astype(np.uint8), erode_kernel).astype(bool)
        cur_use = cur_er if cur_er.any() else cur
        valid_center = cur_use & (~ambiguous)
        if not valid_center.any():
            valid_center = cur_use
        ys_v, xs_v = np.where(valid_center)
        coords = np.stack([xs_v, ys_v], axis=1).astype(np.float32)
        d2 = (coords[:, 0] - cx0) ** 2 + (coords[:, 1] - cy0) ** 2
        nn_idx = int(d2.argmin())
        cx_s, cy_s = coords[nn_idx]
        snapped_centers.append((cx_s, cy_s))
        areas.append(area0)
        dy = cy_s - ys_i
        dx = cx_s - xs_i
        offsets[0, ys_i, xs_i] = (dy / (H / 2.0))
        offsets[1, ys_i, xs_i] = (dx / (W / 2.0))
        offset_mask[ys_i, xs_i] = 1.0
        d = np.sqrt((xs_i - cx_s) ** 2 + (ys_i - cy_s) ** 2)
        R = max(d.max(), 1e-4)
        center_dist[ys_i, xs_i] = np.clip(d / R, 0.0, 1.0)
    center_dist[ambiguous] = 1.0
    center_dist = np.clip(center_dist, 0.0, 1.0)
    valid_spatial_mask = (~ambiguous).astype(np.float32)
    out_h_geo, out_w_geo = int(H // target_stride_geo), int(W // target_stride_geo)
    out_h_sem, out_w_sem = int(H // target_stride_sem), int(W // target_stride_sem)
    val_s2 = cv2.resize(valid_spatial_mask, (out_w_sem, out_h_sem), interpolation=cv2.INTER_NEAREST)
    val_s4 = cv2.resize(valid_spatial_mask, (out_w_geo, out_h_geo), interpolation=cv2.INTER_NEAREST)
    off_mask_s4 = cv2.resize(offset_mask, (out_w_geo, out_h_geo), interpolation=cv2.INTER_NEAREST)
    dist_s4 = cv2.resize(center_dist, (out_w_geo, out_h_geo), interpolation=cv2.INTER_LINEAR)
    offsets_hwc = offsets.transpose(1, 2, 0)
    offsets_s4 = cv2.resize(offsets_hwc, (out_w_geo, out_h_geo), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    if len(snapped_centers) == 0:
        return {
            "centers": torch.zeros((1, out_h_geo, out_w_geo)),
            "center_dist": torch.ones((1, out_h_geo, out_w_geo)),
            "offsets": torch.zeros((2, out_h_geo, out_w_geo)),
            "offset_mask": torch.zeros((1, out_h_geo, out_w_geo)),
            "valid_spatial_mask_s2": torch.zeros((1, out_h_sem, out_w_sem)),
            "valid_spatial_mask_s4": torch.zeros((1, out_h_geo, out_w_geo)),
        }
    snapped_centers_float = np.asarray(snapped_centers, np.float32) / target_stride_geo
    snapped_centers_int = np.round(snapped_centers_float).astype(int)
    areas = np.asarray(areas, np.float32)
    radii_stride = np.sqrt(areas / np.pi) / target_stride_geo
    sigma = radii_stride * sigma_scale
    sigma = np.maximum(sigma, sigma_min)
    dynamic_max = np.maximum(radii_stride * max_sigma_ratio, 0.5)
    sigma = np.minimum(sigma, dynamic_max)
    sigma = np.minimum(sigma, sigma_max)
    if len(snapped_centers_float) >= 2:
        tree = cKDTree(snapped_centers_float)
        dists, _ = tree.query(snapped_centers_float, k=2)
        sigma = np.minimum(sigma, dists[:, 1] * neighbor_clip)
    sigma = np.maximum(sigma, 0.1)
    yy, xx = np.meshgrid(np.arange(out_h_geo), np.arange(out_w_geo), indexing="ij")
    heat = np.zeros((out_h_geo, out_w_geo), np.float32)
    for (cx_i, cy_i), s in zip(snapped_centers_int, sigma):
        g = np.exp(-((xx - cx_i) ** 2 + (yy - cy_i) ** 2) / (2 * s * s))
        heat = np.maximum(heat, g)
    return {
        "centers": torch.from_numpy(heat[None]).float(),
        "center_dist": torch.from_numpy(dist_s4[None]).float(),
        "offsets": torch.from_numpy(offsets_s4).float(),
        "offset_mask": torch.from_numpy(off_mask_s4[None]).float(),
        "valid_spatial_mask_s2": torch.from_numpy(val_s2[None]).float(), 
        "valid_spatial_mask_s4": torch.from_numpy(val_s4[None]).float(),
    }