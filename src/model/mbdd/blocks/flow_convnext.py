import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 1. 大核深度可分离卷积，捕捉空间对齐所需的几何偏差
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.LayerNorm(dim)
        # 2. 1x1 卷积（Pointwise）进行通道扩张
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class flowmlp_convnext(nn.Module):
    def __init__(self, inplane, **kwargs):
        super(flowmlp_convnext, self).__init__()
        # 1. 模仿 ConvNeXt 的 Stem/Block 设计
        # 使用 7x7 深度卷积，极大地增强空间对齐的感知范围
        self.dwconv = nn.Conv2d(inplane, inplane, kernel_size=7, padding=3, groups=inplane)
        self.norm = nn.LayerNorm(inplane)
        
        # 2. 这里的 MLP 结构替代了原先的 ResNetBasicBlock
        self.pwconv1 = nn.Linear(inplane, 4 * inplane) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * inplane, inplane)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C) 以适配 LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # 换回 (N, C, H, W)
        x = input + self.drop(x) # 残差连接
        return x

class Mlp_convnext(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # 最后一步通常不加 act，除非你希望限制特征的非线性表达
        x = self.drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        # 加上 batch_first=True，适配你 (B, L, C) 的数据流
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # 此时 query, key, value 形状均为 (B, L, C)
        attn_output, attn_weights = self.multihead_attn(query, key, value,need_weights=False)
        attn_output = self.dropout(attn_output)
        
        # 残差连接 + LayerNorm
        output = self.norm(attn_output + query)
        return output

class Flow_ConvNeXt(nn.Module):
    def __init__(self, inplane, **kwargs):
        super(Flow_ConvNeXt, self).__init__()
        # 替换为 ConvNeXt 风格的特征提取
        self.flow_feat1 = ConvNeXtBlock(inplane)
        self.flow_feat2 = ConvNeXtBlock(inplane)
        
        # 光流生成层
        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)
        
        # 交叉注意力对齐
        self.channel_align = CrossAttention(inplane)
        self.norm = nn.LayerNorm(inplane)
        self.drop = nn.Dropout(0.1)
        
        # Mlp 部分保持 GELU 风格
        self.mlp = Mlp_convnext(inplane, inplane, inplane)

    def align(self, x, cond):
        b, c, h, w = x.size()
        x, cond = x.reshape(b, -1, c), cond.reshape(b, -1, c)
        x = x + self.drop(self.channel_align(self.norm(x), self.norm(cond), self.norm(cond))) #Query Key Value
        x, cond = x.reshape(b, c, h, w), cond.reshape(b, c, h, w)
        x = x + self.drop(self.mlp(self.norm(x.permute(0, 2, 3, 1)), h, w)).permute(0, 3, 1, 2) #特征从 (B, C, H, W) 变成 (B, H, W, C)
        return x
        b, c, h, w = x.size()
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        cond_flat = cond.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x_norm = self.norm(x_flat)
        cond_norm = self.norm(cond_flat)
        attn_out = self.channel_align(x_norm, cond_norm, cond_norm)
        x_flat = x_flat + self.drop(attn_out)
        x = x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x_permute = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = x + self.drop(self.mlp(self.norm(x_permute), h, w)).permute(0, 3, 1, 2)
        return x 

    def forward(self, x1, x2):
        # 1. 提取用于光流计算的特征
        x1_feat = self.flow_feat1(x1)
        x2_feat = self.flow_feat2(x2)
        
        # 2. 语义与空间对齐
        x2_aligned = self.align(x2_feat, x1_feat)
        
        # 3. 计算光流场
        flow = self.flow_make(torch.cat([x2_aligned, x1_feat], dim=1))
        
        # 4. 变形处理 (Warping)
        size = x1.size()[2:]
        # 注意：这里我们是对原始特征 x2 进行 warp，而不是对提取后的特征进行 warp
        seg_flow_warp2 = self.flow_warp(x2, flow, size)
        
        return seg_flow_warp2, flow

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, out_h, device=input.device),
            torch.linspace(-1, 1, out_w, device=input.device),
            indexing='ij'
        )   
    
        # 2. 拼接成 (H, W, 2) 并扩展到 Batch 维度 (N, H, W, 2)
        grid = torch.stack((grid_x, grid_y), dim=-1).float() # (H, W, 2)
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1) # (N, H, W, 2)

        # 3. 归一化 Flow 并叠加
        # 注意：flow 的维度通常是 (N, 2, H, W)，permute 后变为 (N, H, W, 2)
        # 我们直接除以 (width/2, height/2) 这种逻辑更符合 grid_sample 的 [-1, 1] 定义
        norm = torch.tensor([out_w / 2.0, out_h / 2.0], device=input.device).view(1, 1, 1, 2)

        # 这里的 grid 本身就是 [-1, 1]，flow / norm 转换为相对位移
        v_grid = grid + flow.permute(0, 2, 3, 1) / norm

        # 4. 采样
        # align_corners=False 配合上面的 linspace(-1, 1) 是标准做法
        return F.grid_sample(input, v_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
