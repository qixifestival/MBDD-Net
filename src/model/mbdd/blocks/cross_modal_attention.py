import torch
import torch.nn as nn
from .attent_fuse import AttFuseBlock

class CrossModalFusion(nn.Module):
    """跨模态注意力：用光学引导 SAR 特征进行空间重定位"""
    def __init__(self, dims):
        super().__init__()
        self.opt_combine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(opt_dim * 3, opt_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(opt_dim),
                nn.GELU()
            ) for opt_dim in dims])
        
        self.fusion= nn.ModuleList(
            [GatedFusion(dims[i]) for i in range(3)]+
            [AttFuseBlock(dims[3],dims[3]//2, dims[3])]
        )

        self.drop_layers = nn.ModuleList([
            nn.Dropout2d(p=0.3) for _ in range(4)
        ])

        self.gate_scales = nn.Parameter(torch.ones(4) * 2.0) # 放大差异响应
        self.sar_scales = nn.Parameter(torch.ones(4) * 0.5)  # 抑制 SAR 初期噪声

        
        
    def forward(self, f_pre_opt, f_post_opt, f_sar):
        fused_pyramid = []
        for i in range(4):
            diff = torch.abs(f_pre_opt[i] - f_post_opt[i])
            gate = torch.sigmoid(diff* self.gate_scales[i])

            f_pre_filtered = f_pre_opt[i] * gate
            f_opt_combined = torch.cat([f_pre_filtered, f_post_opt[i],diff], dim=1)
            f_opt_final=self.opt_combine[i](f_opt_combined)
            
            current_sar_scale = torch.clamp(self.sar_scales[i], min=0.2)
            f_sar_scaled = f_sar[i] * current_sar_scale
            fused_feat = self.fusion[i](f_opt_final, f_sar_scaled)
            
            fused_feat = self.drop_layers[i](fused_feat)
            
            fused_pyramid.append(fused_feat)      
        return fused_pyramid
    
class GatedFusion(nn.Module):

    def __init__(self, in_channels,kernel_size=3):
        super().__init__()
        self.norm_x = nn.LayerNorm(in_channels)

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=kernel_size,padding= kernel_size //2),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

        self.channel_adj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1), # 先降维，减少参数，增强泛化
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, 1), # 再升维
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.norm_x(x.permute(0,2,3,1)).permute(0,3,1,2)
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * self.channel_adj(y) + y

        return FG + PG
            
