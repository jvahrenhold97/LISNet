import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights, 
    ConvNeXt_Small_Weights, 
    ConvNeXt_Base_Weights
)


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone for feature extraction at multiple scales.
    
    Extracts features at 4x, 8x, 16x, and 32x downsampling rates.
    """
    def __init__(self, model_type='convnext_small', pretrained=True, freeze_norm=True):
        """Initialize ConvNeXt backbone.
        
        Args:
            model_type (str): Model variant 'convnext_tiny', 'convnext_small', or 'convnext_medium'. Defaults to 'convnext_small'.
            pretrained (bool): Load ImageNet pretrained weights. Defaults to True.
            freeze_norm (bool): Freeze LayerNorm parameters during training. Defaults to True.
        """
        super(ConvNeXtBackbone, self).__init__()
        if model_type == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.convnext_tiny(weights=weights)
            self.out_channels = [96, 192, 384, 768]
        elif model_type == 'convnext_small':
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.convnext_small(weights=weights)
            self.out_channels = [96, 192, 384, 768]
        elif model_type == 'convnext_medium':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.convnext_base(weights=weights)
            self.out_channels = [128, 256, 512, 1024]
        else:
            raise ValueError(f"Modell {model_type} nicht unterstützt.")
        self.stage1 = base_model.features[0:2]
        self.stage2 = base_model.features[2:4]
        self.stage3 = base_model.features[4:6]
        self.stage4 = base_model.features[6:8]
        self.freeze_norm = freeze_norm
        if freeze_norm: 
            self._freeze_layer_norm()

    def _freeze_layer_norm(self):
        """Freeze all LayerNorm parameters in the model."""
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.eval()
                for p in m.parameters(): 
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_norm: 
            self._freeze_layer_norm()
        return self

    def forward(self, x):
        """Extract multi-scale features.
        
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            dict: Feature maps at different scales {'s4': ..., 's8': ..., 's16': ..., 's32': ...}.
        """
        s4 = self.stage1(x)
        s8 = self.stage2(s4)
        s16 = self.stage3(s8)
        s32 = self.stage4(s16)
        return {"s4": s4, "s8": s8, "s16": s16, "s32": s32}


class BiFPNBlock(nn.Module):
    """Bi-directional Feature Pyramid Network block for multi-scale feature fusion.
    
    Fuses features from multiple scales in top-down and bottom-up paths with learned weights.
    """
    def __init__(self, channels, epsilon=1e-4):
        """Initialize BiFPN block.
        
        Args:
            channels (int): Number of channels in features.
            epsilon (float): Small value for numerical stability. Defaults to 1e-4.
        """
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        def conv_block():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True) 
            )
        self.conv_td = nn.ModuleList([conv_block() for _ in range(3)])
        self.conv_out = nn.ModuleList([conv_block() for _ in range(4)])
        self.w_td = nn.Parameter(torch.ones(2, 3))
        self.w_out = nn.Parameter(torch.ones(3, 4))

    def forward(self, inputs):
        """Fuse multi-scale features via top-down and bottom-up paths.
        
        Args:
            inputs (list): [s4, s8, s16, s32] feature maps at different scales.
        
        Returns:
            list: Fused feature maps [s4_out, s8_out, s16_out, s32_out].
        """
        s4, s8, s16, s32 = inputs
        w1 = F.relu(self.w_td) / (torch.sum(F.relu(self.w_td), dim=0) + self.epsilon)
        w2 = F.relu(self.w_out) / (torch.sum(F.relu(self.w_out), dim=0) + self.epsilon)
        s16_td = self.conv_td[0](w1[0,0]*s16 + w1[1,0]*F.interpolate(s32, size=s16.shape[-2:], mode='bilinear', align_corners=False))
        s8_td = self.conv_td[1](w1[0,1]*s8 + w1[1,1]*F.interpolate(s16_td, size=s8.shape[-2:], mode='bilinear', align_corners=False))
        s4_td = self.conv_td[2](w1[0,2]*s4 + w1[1,2]*F.interpolate(s8_td, size=s4.shape[-2:], mode='bilinear', align_corners=False))
        s4_out = self.conv_out[0](w2[0,0]*s4 + w2[1,0]*s4_td)
        s8_out = self.conv_out[1](w2[0,1]*s8 + w2[1,1]*s8_td + w2[2,1]*F.max_pool2d(s4_out, kernel_size=2, stride=2))
        s16_out = self.conv_out[2](w2[0,2]*s16 + w2[1,2]*s16_td + w2[2,2]*F.max_pool2d(s8_out, kernel_size=2, stride=2))
        w2_s32 = w2[:2, 3] / (torch.sum(w2[:2, 3]) + self.epsilon)
        s32_out = self.conv_out[3](w2_s32[0]*s32 + w2_s32[1]*F.max_pool2d(s16_out, kernel_size=2, stride=2))
        return [s4_out, s8_out, s16_out, s32_out]


class BiFPN(nn.Module):
    """Multi-layer BiFPN for progressive multi-scale feature fusion.
    
    Stacks multiple BiFPN blocks to progressively refine and fuse features across scales.
    """
    def __init__(self, in_channels_list, fpn_channels=128, num_layers=2):
        """Initialize BiFPN.
        
        Args:
            in_channels_list (list): Input channel counts for each scale [c_s4, c_s8, c_s16, c_s32].
            fpn_channels (int): Unified channel count for FPN features. Defaults to 128.
            num_layers (int): Number of BiFPN blocks to stack. Defaults to 2.
        """
        super(BiFPN, self).__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, fpn_channels, 1, bias=False), nn.BatchNorm2d(fpn_channels))
            for ch in in_channels_list
        ])
        self.layers = nn.ModuleList([BiFPNBlock(fpn_channels) for _ in range(num_layers)])

    def forward(self, features):
        """Fuse backbone features through BiFPN layers.
        
        Args:
            features (dict): Backbone features {'s4': ..., 's8': ..., 's16': ..., 's32': ...}.
        
        Returns:
            dict: Fused features at unified channel count {'s4': ..., 's8': ..., 's16': ..., 's32': ...}.
        """
        keys = ["s4", "s8", "s16", "s32"]
        s_list = [self.projections[i](features[keys[i]]) for i in range(4)]
        for layer in self.layers:
            s_list = layer(s_list) 
        s4_out, s8_out, s16_out, s32_out = s_list
        return {
            "s4": s4_out,   
            "s8": s8_out,   
            "s16": s16_out,   
            "s32": s32_out    
        }