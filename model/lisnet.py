import torch
import torch.nn as nn
from .funcs import refine_coords_with_offsets, find_peaks_with_voting
from .backbone import ConvNeXtBackbone, BiFPN
from .blocks import CenterHead, GeometryHead, SemanticHead, CenterSeededInstanceDecoder


class LISNet(nn.Module):
    """Leaf Instance Segmentation Network with center-seeded instance decoding.
    
    Combines ConvNeXt backbone, BiFPN decoder, center detection, offset refinement,
    semantic segmentation, and center-seeded instance mask prediction.
    """
    def __init__(
        self, 
        num_classes=2, 
        max_queries=64, 
        center_thresh=0.35, 
        fpn_channels=128,
        decoder_width=256,
        decoder_layers=6, 
        decoder_heads=8, 
        backbone="convnext_small",
        num_def_points=32,
        enable_closing=False,
        enable_ms_dilation=False,
        disable_coordatt=False,
        disable_dropblock=False,
        enable_spatial_bias=False,
    ):
        """Initialize LISNet with backbone and task heads.
        
        Args:
            num_classes (int): Number of semantic classes. Defaults to 2.
            max_queries (int): Maximum number of instance queries. Defaults to 64.
            center_thresh (float): Threshold for center detection during inference. Defaults to 0.35.
            fpn_channels (int): FPN feature channels. Defaults to 128.
            decoder_width (int): Transformer decoder hidden dimension. Defaults to 256.
            decoder_layers (int): Number of transformer decoder layers. Defaults to 6.
            decoder_heads (int): Number of attention heads. Defaults to 8.
            backbone (str): Backbone type ('convnext_tiny', 'convnext_small', etc). Defaults to 'convnext_small'.
            num_def_points (int): Number of deformable sampling points. Defaults to 32.
            enable_closing (bool): Enable morphological closing in instance decoder. Defaults to False.
            enable_ms_dilation (bool): Enable multi-scale dilation. Defaults to False.
            disable_coordatt (bool): Disable coordinate attention. Defaults to False.
            disable_dropblock (bool): Disable DropBlock regularization. Defaults to False.
            enable_spatial_bias (bool): Enable spatial bias in decoder. Defaults to False.
        """
        super().__init__()
        self.max_queries = max_queries
        self.center_thresh = center_thresh
        self.backbone = ConvNeXtBackbone(model_type=backbone, pretrained=True, freeze_norm=True)
        self.decoder = BiFPN(in_channels_list=self.backbone.out_channels, fpn_channels=fpn_channels, num_layers=2)
        self.center_head = CenterHead(
            fpn_channels=fpn_channels, 
            disable_dropblock=disable_dropblock,
            enable_ms_dilation=enable_ms_dilation,
            disable_coordatt=disable_coordatt
        )
        self.geometry_head = GeometryHead(
            fpn_channels=fpn_channels,
            disable_coordatt=disable_coordatt
        )
        self.semantic_head = SemanticHead(
            fpn_channels=fpn_channels, 
            num_classes=num_classes,
            disable_coordatt=disable_coordatt
        )
        self.mask_decoder = CenterSeededInstanceDecoder(
            in_channels=fpn_channels,
            hidden_channels=fpn_channels,
            d_model=decoder_width,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            npoints=num_def_points,
            enable_closing=enable_closing,
            enable_spatial_bias=enable_spatial_bias 
        )

    def forward(self, x, prompt_coords=None):
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            tuple: (out_fields, pred_masks, score_logits, coords_list) where:
                - out_fields (dict): All model outputs including centers, offsets, semantic, masks, scores, coords
                - pred_masks (torch.Tensor): Instance mask predictions
                - score_logits (torch.Tensor): Mask score predictions
                - coords_list (list): Detected center coordinates
        """
        features = self.backbone(x)
        bifpn_features = self.decoder(features)
        s4 = bifpn_features["s4"]
        s8 = bifpn_features["s8"]
        semantic_logits, s2 = self.semantic_head(s4, s8)
        centers = self.center_head(s4, s8)
        offsets, dist = self.geometry_head(s4.detach())
        with torch.no_grad():
            center_prob = torch.sigmoid(centers).detach()
            if prompt_coords is not None:
                norm_coords_list = prompt_coords
                scores_list = [torch.ones(len(c), device=centers.device, dtype=torch.float32) for c in norm_coords_list]
                coords_list = prompt_coords
            else:
                current_thr = 0.0 if self.training else self.center_thresh
                coords_list, scores_list = find_peaks_with_voting(center_prob, offsets, abs_thr=current_thr, max_queries=self.max_queries)
                coords_list = refine_coords_with_offsets(center_prob, coords_list, offsets)
                _, _, H_feat, W_feat = centers.shape 
                norm_coords_list = []
                for i, c in enumerate(coords_list):
                    if c.numel() > 0:
                        y_norm = c[:, 0] / max(H_feat - 1, 1)
                        x_norm = c[:, 1] / max(W_feat - 1, 1)
                        norm_coords_list.append(torch.stack([y_norm, x_norm], dim=1))
                    else:
                        norm_coords_list.append(torch.tensor([[0.5, 0.5]], device=centers.device, dtype=torch.float32))
                        scores_list[i] = torch.tensor([1e-4], device=centers.device, dtype=torch.float32)
        inst = self.mask_decoder(
            feat_s2=s2,
            feat_s4=s4,
            feat_s8=s8,
            coords_list=norm_coords_list,
            scores_list=scores_list,
            max_queries=self.max_queries,
        )
        out_fields = {
            "centers": centers,
            "offsets": offsets,
            "dist": dist,
            "semantic": semantic_logits,
            "pred_masks": inst["pred_masks"],
            "query_valid": inst["query_valid"],
            "score_logits": inst["score_logits"],
            "query_coords": inst["query_coords"],
        } 
        return out_fields, inst["pred_masks"], inst["score_logits"], coords_list

    def trainable_parameters(self):
        """Count number of trainable parameters.
        
        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)