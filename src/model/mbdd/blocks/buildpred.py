import torch
import torch.nn as nn
import torch.nn.functional as F

class BuildingHead(nn.Module):
    def __init__(self, in_ch=256, edge_ch=64, num_classes=1):
        super().__init__()
        
        # 1. 语义特征压缩 (1/4 尺度)
        self.sem_conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # 2. 细节分支处理 (1/2 尺度)
        self.edge_process = nn.Sequential(
            nn.Conv2d(edge_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 3. 核心改动：门控注意力 (让语义特征去“筛选”边缘)
        # 解决“坨”的关键：只有语义认定的地方，才去强化边缘
        self.gate = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        # 4. 边界强化卷积 (使用较大的扩张率或特定结构)
        # 用来把挨在一起的楼“切开”
        self.refine = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1, groups=3, bias=False), # 分组卷积保持独立性
            nn.BatchNorm2d(96),
            nn.GELU(),
            # 引入一个 1x1 卷积专门负责类间竞争
            nn.Conv2d(96, 64, 1), 
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        self.final_cls = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, feat_edge):
        # x: 1/4 语义, feat_edge: 1/2 Stem
        
        # A. 融合 1/4 特征
        feat_1_4 = self.sem_conv(x)
        
        # B. 细节处理与上采样
        feat_edge_proc = self.edge_process(feat_edge)
        feat_1_4_up = F.interpolate(feat_1_4, size=feat_edge.shape[-2:], mode='bilinear', align_corners=True)
        
        # C. 门控细节注入 (关键步骤)
        # 用 1/4 的语义图生成一个 Mask，过滤掉背景区域的边缘噪声
        gate_map = self.gate(feat_1_4_up)
        feat_edge_refined = feat_edge_proc * gate_map
        
        # D. 最终特征聚合
        feat_final = torch.cat([feat_1_4_up, feat_edge_refined], dim=1)
        feat_final = self.refine(feat_final)
        
        return self.final_cls(feat_final)