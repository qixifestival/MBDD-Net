import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=1,low_channels=256):
        super().__init__()
        dilated_rate=[1,4,8]
        # 支路 1: 标准 1x1 卷积，保持局部信息
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        # 支路 2: 3x3 卷积，Dilation=6 (感受野变大)
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=dilated_rate[1], dilation=dilated_rate[1], bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        # 支路 3: 3x3 卷积，Dilation=12 (感受野更大)
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=dilated_rate[2], dilation=dilated_rate[2], bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        #感知全局信息
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # 特征压缩
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128 * 4, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(p=0.1) 
        )

        self.aux_head = nn.Conv2d(256, num_classes, 1)

        # 2. 引入浅层特征融合 
        self.low_level_fuse = nn.Sequential(
            nn.Conv2d(low_channels, 128, 1, bias=False), 
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.fusion_dropout = nn.Dropout2d(p=0.2)

        self.gated_fusion = GatedFusion(in_channels=256,kernel_size=3)

        self.final_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(p=0.2), 

            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x, post_f1,out_build=None):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=True)

        res = self.bottleneck(torch.cat([x1, x2, x3, x4], dim=1))
        aux_out = None
        if self.training:
            aux_out = self.aux_head(res) # 1/16 或 1/8 尺度的初步预测
        
        # 融合浅层高频细节
        res = F.interpolate(res, size=post_f1.size()[2:], mode='bilinear', align_corners=True)
        low_f = self.low_level_fuse(post_f1)
        fused_feat= self.gated_fusion(res, low_f)
        fused_feat = self.fusion_dropout(fused_feat)
        if out_build is not None:
            loc_map = F.interpolate(out_build, size=fused_feat.size()[2:], mode='bilinear', align_corners=True)
            loc_map = torch.sigmoid(loc_map).detach()
            loc_map = loc_map * 0.9 + 0.1
            fused_feat = fused_feat * loc_map
        
        out = self.final_head(fused_feat)

        if self.training:
            # 训练模式下返回两个，用于计算两个 Loss
            return out, aux_out
        else:
            # 验证/测试模式下只返回主输出
            return out
    
class GatedFusion(nn.Module):

    def __init__(self, in_channels,kernel_size=3):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=kernel_size,padding= kernel_size //2),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return FG + PG