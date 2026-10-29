import torch.nn.functional as F
import torch, math
import torch.nn as nn


class DetectionNeck(nn.Module):
    """Detection neck combining depthwise and pointwise convolutions with optional attention.
    
    Processes feature maps with optional multi-scale dilation and coordinate attention.
    """
    def __init__(self, channels=128, enable_ms_dilation=False, disable_coordatt=False):
        """Initialize DetectionNeck.
        
        Args:
            channels (int): Number of input/output channels. Defaults to 128.
            enable_ms_dilation (bool): Enable multi-scale dilation branches. Defaults to False.
            disable_coordatt (bool): Disable coordinate attention. Defaults to False.
        """
        super(DetectionNeck, self).__init__()
        self.enable_ms_dilation = enable_ms_dilation
        self.block1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        if enable_ms_dilation:
            mid_channels = channels // 2
            self.dil_1 = nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, dilation=1, bias=False)
            self.dil_3 = nn.Conv2d(channels, mid_channels, kernel_size=3, padding=3, dilation=3, bias=False)
            self.dil_5 = nn.Conv2d(channels, mid_channels, kernel_size=3, padding=5, dilation=5, bias=False)
            self.mixer = nn.Sequential(
                nn.Conv2d(mid_channels * 3, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.single_scale_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            )
        self.coord_att = CoordAtt(channels) if not disable_coordatt else nn.Identity()

    def forward(self, x):
        out1 = self.block1(x)
        if self.enable_ms_dilation:
            d1 = self.dil_1(out1)
            d3 = self.dil_3(out1)
            d5 = self.dil_5(out1)
            concat_features = torch.cat([d1, d3, d5], dim=1)
            out2 = self.mixer(concat_features)
        else:
            out2 = self.single_scale_conv(out1)
        return self.coord_att(out2 + x)
    

class GeometryNeck(nn.Module):
    """Geometry neck for offset/geometry prediction with refinement and attention."""
    def __init__(self, channels=128, disable_coordatt=False):
        """Initialize GeometryNeck.
        
        Args:
            channels (int): Number of input/output channels. Defaults to 128.
            disable_coordatt (bool): Disable coordinate attention. Defaults to False.
        """
        super(GeometryNeck, self).__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.coord_att = CoordAtt(channels) if not disable_coordatt else nn.Identity()
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.refiner(x)
        x = self.coord_att(x)
        return self.final_conv(x)
    

class PixelNeck(nn.Module):
    """Pixel neck for semantic segmentation combining upsampling and edge refinement."""
    def __init__(self, channels=128, disable_coordatt=False):
        """Initialize PixelNeck.
        
        Args:
            channels (int): Number of input/output channels. Defaults to 128.
            disable_coordatt (bool): Disable coordinate attention. Defaults to False.
        """
        super(PixelNeck, self).__init__()
        self.coord_att = CoordAtt(channels) if not disable_coordatt else nn.Identity()
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.edge_refiner = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            self.coord_att
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_s4, p2):
        fused = torch.cat([x_s4, p2], dim=1)
        x = self.initial_fusion(fused)
        x = self.refinement(x)
        x_up = self.upscale(x)
        refined = self.edge_refiner(x_up)
        return x_up + refined


class CoordAtt(nn.Module):
    """Coordinate Attention for spatial feature recalibration.
    
    Generates channel-wise attention maps from spatial average pooling.
    """
    def __init__(self, channels, reduction=16):
        """Initialize Coordinate Attention.
        
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for bottleneck. Defaults to 16.
        """
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)

    def forward(self, x):
        """Apply coordinate attention.
        
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).
        
        Returns:
            torch.Tensor: Attention-weighted tensor (B, C, H, W).
        """
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))
    

class DropBlock2D(nn.Module):
    """2D DropBlock regularization for spatial feature dropout.
    
    Drops spatially correlated regions instead of individual activations.
    """
    def __init__(self, drop_prob=0.3, block_size=5):
        """Initialize DropBlock2D.
        
        Args:
            drop_prob (float): Probability of dropping a region. Defaults to 0.3.
            block_size (int): Size of dropped square block. Defaults to 5.
        """
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        B, C, H, W = x.shape
        gamma = (self.drop_prob * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask = torch.rand(B, C, H, W, device=x.device, dtype=x.dtype) < gamma
        mask = mask.float()
        padding = self.block_size // 2
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=padding)
        mask = 1.0 - mask
        normalize_scale = mask.numel() / (mask.sum() + 1e-6)
        return x * mask * normalize_scale
    

class CenterHead(nn.Module):
    """Head for predicting center/keypoint heatmaps.
    
    Fuses multi-scale features and outputs single-channel center heatmap.
    """
    def __init__(self, fpn_channels=128, disable_dropblock=False, enable_ms_dilation=False, disable_coordatt=False):
        """Initialize CenterHead.
        
        Args:
            fpn_channels (int): Number of FPN channels. Defaults to 128.
            disable_dropblock (bool): Disable DropBlock regularization. Defaults to False.
            enable_ms_dilation (bool): Enable multi-scale dilation. Defaults to False.
            disable_coordatt (bool): Disable coordinate attention. Defaults to False.
        """
        super(CenterHead, self).__init__()
        self.s8_upsample = nn.ConvTranspose2d(
            in_channels=fpn_channels, 
            out_channels=fpn_channels, 
            kernel_size=2, 
            stride=2
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * 2, fpn_channels, kernel_size=3, padding=1, groups=fpn_channels, bias=False),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.SiLU(inplace=True)
        )
        self.neck = DetectionNeck(fpn_channels, enable_ms_dilation, disable_coordatt)
        self.dropblock = DropBlock2D(drop_prob=0.30, block_size=5) if not disable_dropblock else nn.Identity()
        self.heatmap_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.normal_(self.heatmap_head.weight, std=0.01)
        nn.init.constant_(self.heatmap_head.bias, bias_value)

    def forward(self, s4, s8):
        """Predict center heatmap from multi-scale features.
        
        Args:
            s4 (torch.Tensor): Features at 4x downsampling.
            s8 (torch.Tensor): Features at 8x downsampling.
        
        Returns:
            torch.Tensor: Center heatmap predictions (B, 1, H, W).
        """
        s8_up = self.s8_upsample(s8)
        if s8_up.shape[-2:] != s4.shape[-2:]:
            s8_up = F.interpolate(s8_up, size=s4.shape[-2:], mode="nearest")
        fused = torch.cat([s4, s8_up], dim=1)
        fused_feat = self.fusion_conv(fused)
        feat = self.neck(fused_feat)
        feat = self.dropblock(feat)
        return self.heatmap_head(feat)


class GeometryHead(nn.Module):
    """Head for predicting offset and distance maps for coordinate refinement."""
    def __init__(self, fpn_channels=128, disable_coordatt=False):
        """Initialize GeometryHead.
        
        Args:
            fpn_channels (int): Number of FPN channels. Defaults to 128.
            disable_coordatt (bool): Disable coordinate attention. Defaults to False.
        """
        super(GeometryHead, self).__init__()
        self.add_coords = AddCoords()
        self.neck = GeometryNeck(fpn_channels + 2, disable_coordatt) 
        self.offset_head = nn.Conv2d(fpn_channels + 2, 2, kernel_size=1)
        self.dist_head = nn.Conv2d(fpn_channels + 2, 1, kernel_size=1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, p2):
        """Predict offset and distance maps.
        
        Args:
            p2 (torch.Tensor): Input features (B, C, H, W).
        
        Returns:
            tuple: (offsets, distance) maps for refinement.
        """
        x_coords = self.add_coords(p2)
        feat = self.neck(x_coords)
        offsets = self.offset_head(feat)
        dist = self.dist_head(feat)
        return offsets, dist


class SemanticHead(nn.Module):
    """Head for semantic segmentation prediction.
    
    Multi-scale feature fusion followed by classification head.
    """
    def __init__(self, fpn_channels=128, num_classes=2, disable_coordatt=False):
        """Initialize SemanticHead.
        
        Args:
            fpn_channels (int): Number of FPN channels. Defaults to 128.
            num_classes (int): Number of semantic classes. Defaults to 2.
            disable_coordatt (bool): Disable coordinate attention. Defaults to False.
        """
        super(SemanticHead, self).__init__()
        self.p3_adapter = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.context = MultiScaleContext(fpn_channels, fpn_channels)
        self.neck = PixelNeck(fpn_channels, disable_coordatt)
        self.cls_head = nn.Conv2d(fpn_channels, num_classes + 1, kernel_size=1)

    def forward(self, p2, p3):
        """Predict semantic segmentation logits.
        
        Args:
            p2 (torch.Tensor): Features at 4x downsampling.
            p3 (torch.Tensor): Features at 8x downsampling.
        
        Returns:
            tuple: (logits, refined_features) for semantic segmentation.
        """
        p3_adapted = self.p3_adapter(p3)
        p3_up = F.interpolate(p3_adapted, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        fused_feat = p2 + p3_up
        context_feat = self.context(fused_feat)
        feat_stride2 = self.neck(context_feat, p2)
        logits = self.cls_head(feat_stride2)
        return logits, feat_stride2

class AddCoords(nn.Module):
    """Concatenate coordinate maps to feature tensor.
    
    Adds normalized spatial coordinate channels (x, y) to feature maps.
    """
    def __init__(self):
        """Initialize AddCoords."""
        super().__init__()
        self.register_buffer('grid', None, persistent=False)

    def forward(self, x):
        """Concatenate coordinate maps to features.
        
        Args:
            x (torch.Tensor): Input features (B, C, H, W).
        
        Returns:
            torch.Tensor: Features with coordinate channels (B, C+2, H, W).
        """
        B, _, H, W = x.shape
        device = x.device
        dtype = x.dtype
        if self.grid is None or self.grid.shape[2] != H or self.grid.shape[3] != W:
            delta_y = 1.0 / H
            delta_x = 1.0 / W
            y_coords = torch.linspace(-1.0 + delta_y, 1.0 - delta_y, H, device=device, dtype=dtype)
            x_coords = torch.linspace(-1.0 + delta_x, 1.0 - delta_x, W, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([xx, yy], dim=0)
            self.grid = grid.unsqueeze(0)
        batch_grid = self.grid.expand(B, -1, -1, -1)
        return torch.cat([x, batch_grid], dim=1)


class MultiScaleContext(nn.Module):
    """Multi-scale context aggregation with parallel dilated convolutions."""
    def __init__(self, in_channels, out_channels):
        """Initialize MultiScaleContext.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        mid_c = out_channels // 4
        def make_branch(dilation, is_1x1=False):
            if is_1x1:
                return nn.Sequential(
                    nn.Conv2d(in_channels, mid_c, 1, bias=False),
                    nn.BatchNorm2d(mid_c),
                    nn.ReLU(inplace=True)
                )
            return nn.Sequential(
                nn.Conv2d(in_channels, mid_c, 1, bias=False),
                nn.BatchNorm2d(mid_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, mid_c, (3, 1), padding=(dilation, 0), dilation=(dilation, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, mid_c, (1, 3), padding=(0, dilation), dilation=(1, dilation), bias=False),
                nn.BatchNorm2d(mid_c),
                nn.ReLU(inplace=True)
            )
        self.b1 = make_branch(1, is_1x1=True)
        self.b2 = make_branch(1)
        self.b3 = make_branch(2)
        self.b4 = make_branch(4)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.fusion(feat)
    

class DeformableCrossAttn1Scale(nn.Module):
    """Deformable cross-attention operating at a single scale.
    
    Applies deformable attention to sample relevant regions from memory.
    """
    def __init__(self, d_model=256, n_heads=8, n_points=4):
        """Initialize DeformableCrossAttn1Scale.
        
        Args:
            d_model (int): Model dimension. Defaults to 256.
            n_heads (int): Number of attention heads. Defaults to 8.
            n_points (int): Number of deformable sampling points. Defaults to 4.
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.offset_mlp = nn.Linear(d_model, n_heads * n_points * 2)
        self.weight_mlp = nn.Linear(d_model, n_heads * n_points)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.offset_mlp.weight)
        nn.init.zeros_(self.offset_mlp.bias)
        nn.init.zeros_(self.weight_mlp.weight)
        nn.init.zeros_(self.weight_mlp.bias)

    def forward(self, query, memory_2d, ref_points):
        B, Q, D = query.shape
        _, _, H, W = memory_2d.shape
        q = self.q_proj(query)
        v = self.v_proj(memory_2d)
        offsets = self.offset_mlp(q)
        offsets = offsets.view(B, Q, self.n_heads, self.n_points, 2)
        attn_w = self.weight_mlp(q)
        attn_w = attn_w.view(B, Q, self.n_heads, self.n_points)
        attn_w = F.softmax(attn_w, dim=-1)
        off_x = offsets[..., 0] * (2.0 / max(W - 1, 1))
        off_y = offsets[..., 1] * (2.0 / max(H - 1, 1))
        ref = ref_points.unsqueeze(2).unsqueeze(3)
        ref_x = ref[..., 0]  
        ref_y = ref[..., 1]
        grid_x = ref_x + off_x  
        grid_y = ref_y + off_y
        grid = torch.stack([grid_x, grid_y], dim=-1)
        v = v.view(B, self.n_heads, self.d_head, H, W)
        v = v.reshape(B * self.n_heads, self.d_head, H, W)
        grid_ = grid.permute(0, 2, 1, 3, 4)
        grid_ = grid_.reshape(B * self.n_heads, Q * self.n_points, 1, 2)
        sampled = F.grid_sample(
            v, grid_, mode="bilinear",
            padding_mode="zeros",
            align_corners=False 
        )
        sampled = sampled.squeeze(-1)
        sampled = sampled.view(B, self.n_heads, self.d_head, Q, self.n_points)
        sampled = sampled.permute(0, 3, 1, 4, 2)
        out = (attn_w.unsqueeze(-1) * sampled).sum(dim=3)
        out = out.reshape(B, Q, D)
        return self.out_proj(out)


class RefinementBlock(nn.Module):
    """Refinement block for feature processing with depthwise/pointwise convolutions."""
    def __init__(self, in_channels, out_channels, groups=8):
        """Initialize RefinementBlock.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            groups (int): Number of groups for grouped convolutions. Defaults to 8.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.silu(out + identity)
    

class DeformableDecoderLayer(nn.Module):
    """Transformer decoder layer with deformable cross-attention."""
    def __init__(self, d_model=256, n_heads=8, n_points=12, dim_ff=1024):
        """Initialize DeformableDecoderLayer.
        
        Args:
            d_model (int): Model dimension. Defaults to 256.
            n_heads (int): Number of attention heads. Defaults to 8.
            n_points (int): Number of deformable sampling points. Defaults to 12.
            dim_ff (int): Feedforward dimension. Defaults to 1024.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = DeformableCrossAttn1Scale(d_model=d_model, n_heads=n_heads, n_points=n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.refine_mlp = nn.Linear(d_model, 2) 
        nn.init.zeros_(self.refine_mlp.weight)
        nn.init.zeros_(self.refine_mlp.bias)

    def forward(self, q, memory_2d, ref_points, q_valid_mask=None):
        q2, _ = self.self_attn(q, q, q, key_padding_mask=q_valid_mask)
        q = self.norm1(q + q2)
        ref_points_scaled = ref_points * 2.0 - 1.0
        q2 = self.cross_attn(q, memory_2d, ref_points_scaled)
        q = self.norm2(q + q2)
        q2 = self.ffn(q)
        q = self.norm3(q + q2)
        offsets = self.refine_mlp(q) * 0.1
        new_ref_points = torch.clamp(ref_points + offsets, 0.0, 1.0)
        return q, new_ref_points.detach()


class CenterSeededInstanceDecoder(nn.Module):
    """Center-seeded instance decoder using transformer with deformable attention.
    
    Decodes instance masks from center coordinates using multi-layer deformable transformer.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        d_model=256,
        nhead=8,
        num_layers=4,
        npoints=32,
        enable_closing=False,
        enable_spatial_bias=False,
    ):
        """Initialize CenterSeededInstanceDecoder.
        
        Args:
            in_channels (int): Number of input feature channels.
            hidden_channels (int): Hidden feature dimension.
            d_model (int): Transformer model dimension. Defaults to 256.
            nhead (int): Number of attention heads. Defaults to 8.
            num_layers (int): Number of transformer layers. Defaults to 4.
            npoints (int): Number of deformable sampling points. Defaults to 32.
            enable_closing (bool): Enable morphological closing. Defaults to False.
            enable_spatial_bias (bool): Enable spatial bias in decoding. Defaults to False.
        """
        super().__init__()
        self.pos_dim = hidden_channels // 2 
        dim_feedforward = d_model * 4
        self.mem_proj = RefinementBlock(in_channels + self.pos_dim, d_model)
        self.pix_proj = RefinementBlock(in_channels * 2 + self.pos_dim, hidden_channels)
        self.query_in = nn.Linear(in_channels, d_model, bias=False)
        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=d_model,
                n_heads=nhead,
                n_points=npoints,
                dim_ff=dim_feedforward,
            ) for _ in range(num_layers)
        ])
        self.q_norm = nn.LayerNorm(d_model)
        self.mask_embed_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, hidden_channels),
        )
        nn.init.xavier_uniform_(self.mask_embed_head[-1].weight, gain=1.0)
        self.mask_scale = nn.Parameter(torch.ones(1) * 12.0) 
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1)
        )
        self.spatial_head = nn.Linear(d_model, 4)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.constant_(self.spatial_head.bias, 0.0)
        self.spatial_head.bias.data[2] = -3.0
        self.spatial_head.bias.data[3] = -3.0
        self.null_query = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.null_query, std=0.02)
        self.fuse_logits = nn.Parameter(torch.zeros(3))
        self.pos_query_proj = nn.Sequential(
            nn.Linear(self.pos_dim, d_model), 
            nn.ReLU(), 
            nn.Linear(d_model, d_model)
        )
        self.enable_spatial_bias = enable_spatial_bias
        self.enable_closing = enable_closing

    def _get_sin_cos_pos_embed(self, h, w, dim, device, dtype, cache_key):
        if hasattr(self, cache_key):
            cached = getattr(self, cache_key)
            if cached.shape[-2:] == (h, w) and cached.shape[1] == dim:
                return cached
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij"
        )
        grid_y = (grid_y / max(h - 1, 1)) * 2 * math.pi
        grid_x = (grid_x / max(w - 1, 1)) * 2 * math.pi
        dim_t = torch.arange(dim // 4, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (2 * dim_t / (dim // 4)))
        sin_x = torch.sin(grid_x.unsqueeze(-1) * inv_freq)
        cos_x = torch.cos(grid_x.unsqueeze(-1) * inv_freq)
        sin_y = torch.sin(grid_y.unsqueeze(-1) * inv_freq)
        cos_y = torch.cos(grid_y.unsqueeze(-1) * inv_freq)
        pos_embed = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1).permute(2, 0, 1).unsqueeze(0).detach()
        setattr(self, cache_key, pos_embed)
        return pos_embed
    
    def _get_query_sincos_embed(self, norm_coords, dim=64):
        device, dtype = norm_coords.device, norm_coords.dtype
        dim_t = torch.arange(dim // 4, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (2 * dim_t / (dim // 4)))
        y, x = norm_coords[..., 0], norm_coords[..., 1]
        y_scaled = y * 2 * math.pi
        x_scaled = x * 2 * math.pi
        sin_x = torch.sin(x_scaled.unsqueeze(-1) * inv_freq)
        cos_x = torch.cos(x_scaled.unsqueeze(-1) * inv_freq)
        sin_y = torch.sin(y_scaled.unsqueeze(-1) * inv_freq)
        cos_y = torch.cos(y_scaled.unsqueeze(-1) * inv_freq)
        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
    
    def forward(self, feat_s2, feat_s4, feat_s8, coords_list, scores_list=None, max_queries=64):
        B, C, H1, W1 = feat_s2.shape 
        device, dtype = feat_s2.device, feat_s2.dtype
        target_size = feat_s4.shape[-2:]
        w = torch.softmax(self.fuse_logits, dim=0)
        def resize(tensor):
            return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False) if tensor.shape[-2:] != target_size else tensor
        fused_mem = w[0] * resize(feat_s2) + w[1] * resize(feat_s4) + w[2] * resize(feat_s8)
        pos_mem = self._get_sin_cos_pos_embed(target_size[0], target_size[1], self.pos_dim, device, dtype, "_cache_mem")
        mem_2d = self.mem_proj(torch.cat([fused_mem, pos_mem.expand(B, -1, -1, -1)], dim=1))
        pos_pix = self._get_sin_cos_pos_embed(H1, W1, self.pos_dim, device, dtype, "_cache_pix")
        s8_up = F.interpolate(feat_s8, size=(H1, W1), mode="bilinear", align_corners=False)
        pix_2d = self.pix_proj(torch.cat([feat_s2, s8_up, pos_pix.expand(B, -1, -1, -1)], dim=1))
        padded_coords = torch.zeros((B, max_queries, 2), device=device, dtype=dtype)
        q_valid = torch.zeros((B, max_queries), device=device, dtype=torch.bool)
        for b, c in enumerate(coords_list):
            n = min(c.shape[0], max_queries)
            if n > 0:
                padded_coords[b, :n] = c[:n]
                q_valid[b, :n] = True
        cy, cx = padded_coords[..., 0], padded_coords[..., 1]
        sample_grid_neg1_to_1 = torch.stack([cx * 2.0 - 1.0, cy * 2.0 - 1.0], dim=-1)
        feat_p1_smoothed = F.avg_pool2d(feat_s2, kernel_size=5, stride=1, padding=2)
        sampled_feats = F.grid_sample(feat_p1_smoothed, sample_grid_neg1_to_1.unsqueeze(2), align_corners=False).squeeze(-1).permute(0, 2, 1)
        q_pos = self.pos_query_proj(self._get_query_sincos_embed(padded_coords, dim=self.pos_dim))
        q = self.query_in(sampled_feats) + q_pos
        q = torch.where(q_valid.unsqueeze(-1), q, self.null_query.view(1, 1, -1))
        ignore_mask = ~q_valid 
        current_ref_points = torch.stack([cx, cy], dim=-1)
        for layer in self.layers:
            q, current_ref_points = layer(
                q, 
                mem_2d, 
                current_ref_points, 
                q_valid_mask=ignore_mask
            )
        q = self.q_norm(q)
        score_logits = self.score_head(q.detach()).squeeze(-1)
        mask_embed = self.mask_embed_head(q)
        mask_embed_norm = F.normalize(mask_embed, p=2, dim=-1)
        pix_2d_norm = F.normalize(pix_2d, p=2, dim=1)
        pred_masks_sim = torch.einsum("bqd,bdhw->bqhw", mask_embed_norm, pix_2d_norm)
        grid_y, grid_x = torch.meshgrid(torch.arange(H1, device=device), torch.arange(W1, device=device), indexing="ij")
        grid_y = (grid_y.to(dtype) / max(H1 - 1, 1)) * 2.0 - 1.0
        grid_x = (grid_x.to(dtype) / max(W1 - 1, 1)) * 2.0 - 1.0
        rel_y = grid_y.view(1, 1, H1, W1) - sample_grid_neg1_to_1[..., 1].view(B, max_queries, 1, 1)
        rel_x = grid_x.view(1, 1, H1, W1) - sample_grid_neg1_to_1[..., 0].view(B, max_queries, 1, 1)
        spatial_params = self.spatial_head(q)
        p_y = torch.tanh(spatial_params[..., 0, None, None]) * 0.15    
        p_x = torch.tanh(spatial_params[..., 1, None, None]) * 0.15
        p_dist_y = -F.softplus(spatial_params[..., 2, None, None]) * 2.0 
        p_dist_x = -F.softplus(spatial_params[..., 3, None, None]) * 2.0
        spatial_bias = p_y * rel_y + p_x * rel_x + p_dist_y * (rel_y**2) + p_dist_x * (rel_x**2)
        temperature = torch.clamp(self.mask_scale, min=1.0, max=20.0)
        pred_masks_raw = (pred_masks_sim + spatial_bias) * temperature if self.enable_spatial_bias else pred_masks_sim * temperature
        if self.enable_closing:
            m_dilated = F.max_pool2d(pred_masks_raw, kernel_size=3, stride=1, padding=1)
            pred_masks = -F.max_pool2d(-m_dilated, kernel_size=3, stride=1, padding=1)
        else:
            pred_masks = pred_masks_raw
        padding_val = -100.0 
        invalid = (~q_valid).view(B, max_queries, 1, 1)
        pred_masks = pred_masks.masked_fill(invalid, padding_val)
        score_logits = score_logits.masked_fill(~q_valid, -10000.0)
        return {
            "pred_masks": pred_masks,
            "query_valid": q_valid,
            "score_logits": score_logits,
            "query_coords": padded_coords,
        }