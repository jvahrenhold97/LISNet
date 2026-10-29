import torch.nn.functional as F
import torch
import torch.nn as nn


def norm_layer(ch, kind="bn"):
    """Create a normalization layer.
    
    Args:
        ch (int): Number of channels.
        kind (str): Normalization type 'bn' for BatchNorm or 'gn' for GroupNorm. Defaults to 'bn'.
    
    Returns:
        nn.Module: Normalization layer.
    """
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    else:
        return nn.GroupNorm(8, ch)
    

@torch.no_grad()
def find_peaks_adaptive(prob, abs_thr=0.35, max_queries=64, radius=3, scale=False):
    """Find local maxima peaks in a probability map using max pooling.
    
    Args:
        prob (torch.Tensor): Probability heatmap (B, 1, H, W).
        abs_thr (float): Absolute probability threshold. Defaults to 0.35.
        max_queries (int): Maximum number of peaks to return. Defaults to 64.
        radius (int): Local maxima kernel radius. Defaults to 3.
        scale (bool): Whether to scale coordinates by 2. Defaults to False.
    
    Returns:
        tuple: (coords_list, scores_list) where each is a list of B tensors.
    """
    B, _, H, W = prob.shape
    device = prob.device
    local_max = F.max_pool2d(prob, kernel_size=2 * radius + 1, stride=1, padding=radius)
    peak_mask = (prob == local_max) & (prob >= abs_thr)
    scores_flat = (prob * peak_mask.float()).view(B, -1)
    K = min(max_queries, H * W) 
    topk_scores, topk_idx = torch.topk(scores_flat, K, dim=1)
    topk_ys = (topk_idx // W).float()
    topk_xs = (topk_idx % W).float()
    factor = 2.0 if scale else 1.0
    coords_list = []
    scores_list = []
    for b in range(B):
        valid = topk_scores[b] >= abs_thr
        if not valid.any():
            coords_list.append(torch.empty((0, 2), device=device))
            scores_list.append(torch.empty((0,), device=device))
            continue
        final_ys = topk_ys[b][valid]
        final_xs = topk_xs[b][valid]
        final_scores = topk_scores[b][valid]
        final_coords = torch.stack([final_ys, final_xs], dim=1) * factor
        coords_list.append(final_coords)
        scores_list.append(final_scores)
    return coords_list, scores_list


@torch.no_grad()
def find_peaks_with_voting(prob, offset, abs_thr=0.2, max_queries=128, radius=2):
    """Find peaks using voting with offset refinement and NMS.
    
    Detects local maxima, refines positions using offset maps, and applies voting-based NMS.
    
    Args:
        prob (torch.Tensor): Probability heatmap (B, 1, H, W).
        offset (torch.Tensor): Offset maps for coordinate refinement (B, 2, H, W).
        abs_thr (float): Absolute probability threshold. Defaults to 0.2.
        max_queries (int): Maximum number of peaks per image. Defaults to 128.
        radius (int): NMS kernel radius. Defaults to 2.
    
    Returns:
        tuple: (coords_list, scores_list) lists of B coordinate and score tensors.
    """
    B, _, H, W = prob.shape
    device = prob.device
    local_max = F.max_pool2d(prob, kernel_size=2 * radius + 1, stride=1, padding=radius)
    peak_mask = (prob == local_max) & (prob >= abs_thr)
    batch_idx, _, y, x = torch.where(peak_mask)
    if batch_idx.numel() == 0:
        return [torch.empty((0, 2), device=device)] * B, [torch.empty((0,), device=device)] * B
    scores = prob[batch_idx, 0, y, x]
    off_y = offset[batch_idx, 0, y, x] * (H / 2.0)
    off_x = offset[batch_idx, 1, y, x] * (W / 2.0)
    ry = torch.round(y.float() + off_y).long().clamp(0, H - 1)
    rx = torch.round(x.float() + off_x).long().clamp(0, W - 1)
    new_scores = prob[batch_idx, 0, ry, rx]
    better_move = new_scores > scores
    final_y = torch.where(better_move, ry, y)
    final_x = torch.where(better_move, rx, x)
    final_scores = torch.where(better_move, new_scores, scores)
    sorted_indices = torch.argsort(final_scores) 
    b_s = batch_idx[sorted_indices]
    ry_s = final_y[sorted_indices]
    rx_s = final_x[sorted_indices]
    sc_s = final_scores[sorted_indices]
    voting_map = torch.zeros_like(prob)
    voting_map[b_s, 0, ry_s, rx_s] = sc_s
    nms_radius = radius
    final_max = F.max_pool2d(voting_map, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius)
    final_mask = (voting_map == final_max) & (voting_map > 0)
    coords_list = []
    scores_list = []
    for b in range(B):
        b_mask = final_mask[b, 0]
        fy, fx = torch.where(b_mask)
        fs = voting_map[b, 0, fy, fx]
        if fs.numel() > max_queries:
            fs, topk_idx = torch.topk(fs, max_queries)
            fy, fx = fy[topk_idx], fx[topk_idx]
        coords_list.append(torch.stack([fy.float(), fx.float()], dim=1))
        scores_list.append(fs)
    return coords_list, scores_list


@torch.no_grad()
def refine_coords_with_offsets(center_prob, coords_list, offset_map, max_shift=20.0):
    """Refine center coordinates using offset maps and keep improvements via voting.
    
    For each coordinate, samples offset map to refine position. Only accepts refinement
    if probability at new location improves.
    
    Args:
        center_prob (torch.Tensor): Center probability heatmap (B, 1, H, W).
        coords_list (list): List of B coordinate tensors (N_i, 2).
        offset_map (torch.Tensor): Offset refinement maps (B, 2, H, W).
        max_shift (float): Maximum allowed pixel shift. Defaults to 20.0.
    
    Returns:
        list: Refined coordinate tensors for each image in batch.
    """
    B, _, H, W = offset_map.shape
    device = offset_map.device
    max_q = max([c.shape[0] for c in coords_list])
    if max_q == 0:
        return coords_list
    padded_coords = torch.zeros((B, max_q, 2), device=device)
    mask = torch.zeros((B, max_q), dtype=torch.bool, device=device)
    for b, c in enumerate(coords_list):
        n = c.shape[0]
        if n > 0:
            padded_coords[b, :n] = c
            mask[b, :n] = True
    x_norm = (padded_coords[..., 1] / (W - 1)) * 2.0 - 1.0
    y_norm = (padded_coords[..., 0] / (H - 1)) * 2.0 - 1.0
    grid_old = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)
    sampled_offsets = F.grid_sample(offset_map, grid_old, mode='bilinear', align_corners=False)
    old_probs = F.grid_sample(center_prob, grid_old, mode='bilinear', align_corners=False)
    dy_pixel = (sampled_offsets[:, 0, 0, :] * (H / 2.0)).clamp(-max_shift, max_shift)
    dx_pixel = (sampled_offsets[:, 1, 0, :] * (W / 2.0)).clamp(-max_shift, max_shift)
    y_refined = (padded_coords[..., 0] + dy_pixel).clamp(0, H - 1)
    x_refined = (padded_coords[..., 1] + dx_pixel).clamp(0, W - 1)
    x_ref_norm = (x_refined / (W - 1)) * 2.0 - 1.0
    y_ref_norm = (y_refined / (H - 1)) * 2.0 - 1.0
    grid_new = torch.stack([x_ref_norm, y_ref_norm], dim=-1).unsqueeze(1)
    new_probs = F.grid_sample(center_prob, grid_new, mode='bilinear', align_corners=False)
    better_move = new_probs > old_probs
    final_y = torch.where(better_move[:, 0, 0, :], y_refined, padded_coords[..., 0])
    final_x = torch.where(better_move[:, 0, 0, :], x_refined, padded_coords[..., 1])
    refined_list = []
    for b in range(B):
        n = coords_list[b].shape[0]
        refined_list.append(torch.stack([final_y[b, :n], final_x[b, :n]], dim=1))
    return refined_list