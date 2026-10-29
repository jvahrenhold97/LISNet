import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class LISNetLoss(nn.Module):
    """Multi-task loss function for LISNet combining center, offset, semantic, and instance losses.
    
    Computes losses for center heatmap detection, offset refinement, semantic segmentation,
    and instance mask prediction with Hungarian matching.
    """
    def __init__(
        self,
        eps=1e-4,
        mask_fp_penalty_weight=0.7,
        mask_fn_penalty_weight=0.3,
        w_score_neg=0.5,
        w_center=1.5,
        w_offset=1.0,
        w_mask=5.0,
        w_score=2.0,
        w_semantic=1.0,
    ):
        """Initialize LISNetLoss with component weights and parameters.
        
        Args:
            eps (float): Small epsilon for numerical stability. Defaults to 1e-4.
            mask_fp_penalty_weight (float): Weight for false positive penalty in mask loss. Defaults to 0.7.
            mask_fn_penalty_weight (float): Weight for false negative penalty in mask loss. Defaults to 0.3.
            w_score_neg (float): Weight for negative score predictions. Defaults to 0.5.
            w_center (float): Weight for center loss in total loss. Defaults to 1.5.
            w_offset (float): Weight for offset loss. Defaults to 1.0.
            w_mask (float): Weight for instance mask loss. Defaults to 5.0.
            w_score (float): Weight for mask score loss. Defaults to 2.0.
            w_semantic (float): Weight for semantic segmentation loss. Defaults to 1.0.
        """
        super().__init__()
        self.eps = eps
        self.mask_fp_penalty_weight = mask_fp_penalty_weight
        self.mask_fn_penalty_weight = mask_fn_penalty_weight
        self.w_score_neg = w_score_neg
        self.w_center = w_center
        self.w_offset = w_offset
        self.w_mask = w_mask
        self.w_score = w_score
        self.w_semantic = w_semantic

    def semantic_loss(self, logits, target, valid_mask):
        """Compute semantic segmentation loss combining focal loss and Dice loss.
        
        Args:
            logits (torch.Tensor): Semantic logits (B, C, H, W).
            target (torch.Tensor): Ground truth semantic labels (B, H, W).
            valid_mask (torch.Tensor): Validity mask (B, 1, H, W).
        
        Returns:
            torch.Tensor: Combined focal + Dice loss.
        """
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
        ce_loss = F.cross_entropy(logits, target, ignore_index=255, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2.0 * ce_loss)
        probs = F.softmax(logits, dim=1)[:, 1]
        gt_leaf = (target == 1).float()
        m = valid_mask.squeeze(1) * (target != 255).float()
        inter = (probs * gt_leaf * m).sum(dim=(1, 2))
        union = (probs * m).sum(dim=(1, 2)) + (gt_leaf * m).sum(dim=(1, 2))
        dice_loss = 1.0 - (2. * inter + self.eps) / (union + self.eps)
        total_sem = (focal_loss * m).sum() / (m.sum() + self.eps) + dice_loss.mean()
        return total_sem

    def center_loss(self, logits, target, valid_spatial_mask, pos_alpha = 2.0, neg_alpha = 2.0, beta = 4.0, eps = 1e-4):
        """Compute focal loss for center heatmap detection.
        
        Applies weighted focal loss with different weighting for positive and negative pixels.
        
        Args:
            logits (torch.Tensor): Center heatmap logits (B, 1, H, W).
            target (torch.Tensor): Ground truth center heatmap (B, 1, H, W).
            valid_spatial_mask (torch.Tensor): Spatial validity mask.
            pos_alpha (float): Focus exponent for positive samples. Defaults to 2.0.
            neg_alpha (float): Focus exponent for negative samples. Defaults to 2.0.
            beta (float): Power term for negative sample weighting. Defaults to 4.0.
            eps (float): Numerical stability epsilon. Defaults to 1e-4.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        logits = logits.float()
        target = target.float()
        if valid_spatial_mask.shape[-2:] != logits.shape[-2:]:
            valid_spatial_mask = F.interpolate(
                valid_spatial_mask, 
                size=logits.shape[-2:], 
                mode="nearest"
            )
        prob = torch.sigmoid(logits)
        prob = torch.clamp(prob, min=eps, max=1.0 - eps)
        is_pos = (target >= 0.999).float() * valid_spatial_mask
        is_neg = (target < 0.999).float() * valid_spatial_mask
        pos_loss = -torch.pow(1.0 - prob, pos_alpha) * torch.log(prob) * is_pos
        neg_weights = torch.pow(1.0 - target, beta)
        neg_loss = -neg_weights * torch.pow(prob, neg_alpha) * torch.log(1.0 - prob) * is_neg
        num_pos = torch.clamp(is_pos.sum(), min=1.0)
        pos_sum = pos_loss.sum()
        neg_sum = neg_loss.sum()
        total_loss = (pos_sum + neg_sum) / num_pos
        return total_loss

    def offset_loss(self, pred_off, tgt_off, mask, valid_spatial_mask, center_dist):
        """Compute offset refinement loss for center coordinate refinement.
        
        Computes L1 loss separately for foreground and background with boundary weighting.
        
        Args:
            pred_off (torch.Tensor): Predicted offsets (B, 2, H, W).
            tgt_off (torch.Tensor): Target offsets (B, 2, H, W).
            mask (torch.Tensor): Foreground mask (B, 1, H, W).
            valid_spatial_mask (torch.Tensor): Spatial validity mask (B, 1, H, W).
            center_dist (torch.Tensor): Distance from center for boundary weighting.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        active_valid = valid_spatial_mask.detach()
        if mask.sum() == 0: 
            return (pred_off.abs() * active_valid).sum() / (active_valid.sum() * 2.0 + self.eps)
        active_fg_mask = mask.detach() * active_valid
        diff = (pred_off - tgt_off.detach()).abs()
        boundary_weight = 1.0 + 2.0 * center_dist.detach()
        fg_loss = (diff * active_fg_mask * boundary_weight).sum() / (active_fg_mask.sum() * 2.0 + self.eps)
        active_bg_mask = (1.0 - mask.detach()) * active_valid
        bg_loss = (pred_off.abs() * active_bg_mask).sum() / (active_bg_mask.sum() * 2.0 + self.eps)
        return fg_loss + 0.1 * bg_loss

    def _get_boundary(self, mask, kernel_size=5):
        """Extract boundary map using max pooling and morphological operations.
        
        Args:
            mask (torch.Tensor): Binary mask tensor.
            kernel_size (int): Kernel size for dilation/erosion. Defaults to 5.
        
        Returns:
            torch.Tensor: Binary boundary map.
        """
        pad = kernel_size // 2
        dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
        erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
        return dilation - erosion
    
    @torch.no_grad()
    def _hungarian_cost(self, pred_masks_logits, gt_masks, q_coords, gt_coords):
        """Compute cost matrix for Hungarian matching between predicted and ground truth masks.
        
        Cost combines Dice loss, recall loss, and spatial distance.
        
        Args:
            pred_masks_logits (torch.Tensor): Predicted mask logits (Q, H, W).
            gt_masks (torch.Tensor): Ground truth masks (G, H, W).
            q_coords (torch.Tensor): Query center coordinates (Q, 2).
            gt_coords (torch.Tensor): Ground truth center coordinates (G, 2).
        
        Returns:
            np.ndarray: Cost matrix (Q, G).
        """
        if pred_masks_logits.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.unsqueeze(1), 
                size=pred_masks_logits.shape[-2:], 
                mode="nearest"
            ).squeeze(1)
        probs = torch.sigmoid(pred_masks_logits)
        probs_flat = probs.flatten(1)
        gt_flat = gt_masks.flatten(1)
        inter = torch.matmul(probs_flat, gt_flat.T)
        area_pred = probs_flat.sum(1).unsqueeze(1)  
        area_gt = gt_flat.sum(1).unsqueeze(0)       
        union = area_pred + area_gt
        dice_cost = 1.0 - (2 * inter + self.eps) / (union + self.eps)
        recall_cost = 1.0 - (inter / (area_gt + self.eps))
        dist_cost = torch.cdist(q_coords, gt_coords, p=2) 
        cost = dice_cost + recall_cost + 10.0 * dist_cost
        return torch.nan_to_num(cost, nan=1.0).cpu().numpy()
    
    def instance_mask_loss(self, pred, gt_instance, valid_spatial_mask):
        """Compute instance segmentation loss using Hungarian matching.
        
        Performs bipartite matching between predictions and ground truth instances,
        then computes focal, Dice, and boundary IoU losses.
        
        Args:
            pred (dict): Predictions with 'pred_masks', 'score_logits', 'query_valid', 'query_coords'.
            gt_instance (torch.Tensor): Ground truth instance maps (B, H, W).
            valid_spatial_mask (torch.Tensor): Spatial validity mask (B, 1, H, W).
        
        Returns:
            tuple: (mask_loss, score_loss, {'mask_core': core_mask_loss})
        """
        p_masks_all = pred["pred_masks"]
        s_logits_all = pred["score_logits"]
        q_valid_all = pred["query_valid"]
        q_coords_all = pred["query_coords"]
        B, Q, H, W = p_masks_all.shape
        device = p_masks_all.device
        m_loss = torch.tensor(0.0, device=device)
        s_loss = torch.tensor(0.0, device=device)
        total_inst = 0
        for b in range(B):
            v_idx = q_valid_all[b]
            if not v_idx.any():
                continue
            inst = gt_instance[b]
            gt_ids = torch.unique(inst)
            gt_ids = gt_ids[gt_ids > 0]
            p_masks = p_masks_all[b, v_idx]   
            s_logits = s_logits_all[b, v_idx] 
            m_valid = valid_spatial_mask[b]   
            q_coords = q_coords_all[b, v_idx]
            if gt_ids.numel() == 0:
                with torch.no_grad():
                    p_bin = (torch.sigmoid(p_masks) > 0.5).float()
                    valid_area = (p_bin * m_valid).sum(dim=(1, 2))
                    total_area = p_bin.sum(dim=(1, 2)) + self.eps
                    is_valid_query = (valid_area / total_area) >= 0.5
                valid_logits = s_logits[is_valid_query]
                if len(valid_logits) > 0:
                    s_loss += F.l1_loss(torch.sigmoid(valid_logits), torch.zeros_like(valid_logits))
                continue
            gt_masks = (inst.unsqueeze(0) == gt_ids.unsqueeze(1).unsqueeze(2)).float()
            G, H_m, W_m = gt_masks.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(H_m, device=device), torch.arange(W_m, device=device), indexing="ij")
            grid_y = grid_y.float() / max(H_m - 1, 1)
            grid_x = grid_x.float() / max(W_m - 1, 1)
            gt_area = gt_masks.sum(dim=(1, 2)) + self.eps
            gt_cy = (gt_masks * grid_y.unsqueeze(0)).sum(dim=(1, 2)) / gt_area
            gt_cx = (gt_masks * grid_x.unsqueeze(0)).sum(dim=(1, 2)) / gt_area
            gt_coords = torch.stack([gt_cy, gt_cx], dim=1)
            cost = self._hungarian_cost(p_masks, gt_masks, q_coords, gt_coords)
            row, col = linear_sum_assignment(cost)
            if len(row) > 0:
                m_pred = p_masks[row]
                m_gt = gt_masks[col]
                probs = torch.sigmoid(m_pred)
                p_t = probs * m_gt + (1 - probs) * (1 - m_gt)
                focal_weight = (1 - p_t) ** 2.0
                alpha = 0.25 
                alpha_weight = alpha * m_gt + (1 - alpha) * (1 - m_gt)
                bce = F.binary_cross_entropy_with_logits(m_pred, m_gt, reduction='none')
                focal_loss = (alpha_weight * focal_weight * bce).mean(dim=(1, 2)).sum()
                inter = (probs * m_gt).sum(dim=(1, 2))
                union = probs.sum(dim=(1, 2)) + m_gt.sum(dim=(1, 2)) + self.eps
                dice_loss_inst = (1.0 - (2.0 * inter) / union).sum()
                gt_b = self._get_boundary(m_gt.unsqueeze(1), kernel_size=3).squeeze(1)
                pred_b = self._get_boundary(probs.unsqueeze(1), kernel_size=3).squeeze(1)
                b_region = torch.clamp(gt_b + pred_b, 0.0, 1.0) 
                b_inter = (probs * m_gt * b_region).sum(dim=(1, 2))
                b_union = (probs * b_region).sum(dim=(1, 2)) + (m_gt * b_region).sum(dim=(1, 2)) - b_inter
                boundary_iou_loss = (1.0 - (b_inter + self.eps) / (b_union + self.eps)).sum()
                m_loss += (2.0 * focal_loss + 2.0 * dice_loss_inst + 0.5 * boundary_iou_loss)
                total_inst += len(row)
                with torch.no_grad():
                    bin_pred = (probs > 0.5).float() 
                    s_inter = (bin_pred * m_gt * m_valid).sum(dim=(1, 2))
                    s_union = (bin_pred * m_valid).sum(dim=(1, 2)) + (m_gt * m_valid).sum(dim=(1, 2)) - s_inter
                    target_score = (s_inter / (s_union + self.eps)).clamp(0.0, 1.0)
                pred_iou = torch.sigmoid(s_logits[row])
                s_loss += F.l1_loss(pred_iou, target_score)
        avg_m = m_loss / max(total_inst, 1)
        avg_s = s_loss / B
        return avg_m, avg_s, {"mask_core": avg_m.detach()}

    def forward(self, pred, gt):
        """Compute all losses and return weighted total loss.
        
        Args:
            pred (dict): Prediction dict with all model outputs.
            gt (dict): Ground truth dict with all targets.
        
        Returns:
            dict: Dictionary with individual losses ('center', 'offsets', 'semantic', 'mask', 'score', 'mask_core', 'total').
        """
        losses = {}
        s2_valid_mask = gt["valid_spatial_mask_s2"]
        s4_valid_mask = gt["valid_spatial_mask_s4"]
        losses["center"] = self.center_loss(pred["centers"], gt["centers"], s4_valid_mask)
        losses["offsets"] = self.offset_loss(pred["offsets"], gt["offsets"], gt["offset_mask"], s4_valid_mask, gt["center_dist"])
        losses["semantic"] = self.semantic_loss(pred["semantic"], gt["semantic"], s2_valid_mask)
        if "instance" in gt and "pred_masks" in pred:
            m_loss, s_loss, aux = self.instance_mask_loss(pred, gt["instance"], s2_valid_mask)
            losses["mask"], losses["score"], losses["mask_core"] = m_loss, s_loss, aux["mask_core"]
        else:
            zero = losses["center"].new_tensor(0.0)
            losses["mask"] = losses["score"] = losses["mask_core"] = zero
        losses["total"] = (
            self.w_center * losses["center"] + 
            self.w_offset * losses["offsets"] + 
            self.w_mask * losses["mask"] + 
            self.w_score * losses["score"] +
            self.w_semantic * losses["semantic"]
        )
        return losses