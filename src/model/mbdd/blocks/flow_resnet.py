import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBasicBlock(nn.Module):
    """ResNet基本块"""
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out

class ResNetBottleneck(nn.Module):
    """ResNet瓶颈块"""
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.gelu = nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out

class flowmlp_resnet(nn.Module):
    def __init__(self, inplane, **kwargs):
        super(flowmlp_resnet, self).__init__()
        self.gelu = nn.GELU()
        self.conv = nn.Conv2d(inplane, inplane, 3, 1, 1, bias=True, groups=inplane)
        self.res_block = ResNetBasicBlock(inplane, inplane)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block(x)
        x = self.gelu(x)
        return x

#多层感知机（输入+隐藏+输出）
class Mlp_resnet(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(
            query,
            key,
            value,
            need_weights=False,
        )
        attn_output = self.dropout(attn_output)
        output = self.norm(attn_output + query)
        return output

class FlowN_ResNet(nn.Module):
    def __init__(self, inplane, window_size=8, **kwargs):
        """
        implementation of FlowN based on ResNet
        :param inplane: channels of input tensor
        """
        super(FlowN_ResNet, self).__init__()
        self.flowmlp1 = flowmlp_resnet(inplane, **kwargs)
        self.flowmlp2 = flowmlp_resnet(inplane, **kwargs)
        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)
        self.channel_align = CrossAttention(inplane)
        self.mlp = Mlp_resnet(inplane, inplane, inplane, act_layer=nn.GELU, drop=0., **kwargs)
        self.drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(inplane)
        self.window_size = window_size

    def _window_partition(self, x):
        b, c, h, w = x.shape
        window_size = self.window_size
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        padded_h, padded_w = h + pad_h, w + pad_w
        windows = x.reshape(
            b,
            c,
            padded_h // window_size,
            window_size,
            padded_w // window_size,
            window_size,
        )
        windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = windows.reshape(-1, window_size * window_size, c)
        return windows, (b, c, h, w, padded_h, padded_w)

    def _window_reverse(self, windows, shape):
        b, c, h, w, padded_h, padded_w = shape
        window_size = self.window_size
        x = windows.reshape(
            b,
            padded_h // window_size,
            padded_w // window_size,
            window_size,
            window_size,
            c,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(b, c, padded_h, padded_w)
        return x[:, :, :h, :w]

    #基于cond调整x，使用交叉注意力机制、mlp对齐
    def align(self, x, cond):
        if x.shape != cond.shape:
            raise ValueError(f"Expected matching feature shapes, got {x.shape} and {cond.shape}")

        b, c, h, w = x.size()
        x_windows, window_shape = self._window_partition(x)
        cond_windows, _ = self._window_partition(cond)
        x_windows = x_windows + self.drop(
            self.channel_align(
                self.norm(x_windows),
                self.norm(cond_windows),
                self.norm(cond_windows),
            )
        )
        x = self._window_reverse(x_windows, window_shape)
        x = x + self.drop(self.mlp(self.norm(x.permute(0, 2, 3, 1)), h, w)).permute(0, 3, 1, 2) #特征从 (B, C, H, W) 变成 (B, H, W, C)
        return x

    def forward(self, x1, x2):
        # channel alignment
        x1 = self.flowmlp1(x1)
        x2 = self.flowmlp2(x2)
        size = x1.size()[2:]
        x2 = self.align(x2, x1) #语义对齐
        flow = self.flow_make(torch.cat([x2, x1], dim=1)) #生成光流场（空间对齐）
        # warping
        seg_flow_warp2 = self.flow_warp(x2, flow, size)
        return seg_flow_warp2, flow

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        if (h, w) != (out_h, out_w):
            raise ValueError(f"Input spatial size {(h, w)} does not match output size {(out_h, out_w)}")
        if flow.shape != (n, 2, out_h, out_w):
            raise ValueError(f"Unexpected flow shape {flow.shape}")

        grid_y, grid_x = torch.meshgrid(
            (torch.arange(out_h, device=input.device, dtype=flow.dtype) + 0.5) * (2.0 / out_h) - 1.0,
            (torch.arange(out_w, device=input.device, dtype=flow.dtype) + 0.5) * (2.0 / out_w) - 1.0,
            indexing='ij'
        )
    
        # 2. 拼接成 (H, W, 2) 并扩展到 Batch 维度 (N, H, W, 2)
        grid = torch.stack((grid_x, grid_y), dim=-1) # (H, W, 2)
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1) # (N, H, W, 2)

        # 3. 归一化 Flow 并叠加
        # 注意：flow 的维度通常是 (N, 2, H, W)，permute 后变为 (N, H, W, 2)
        # 我们直接除以 (width/2, height/2) 这种逻辑更符合 grid_sample 的 [-1, 1] 定义
        norm = torch.tensor([out_w / 2.0, out_h / 2.0], device=input.device).view(1, 1, 1, 2)

        # 这里的 grid 本身就是 [-1, 1]，flow / norm 转换为相对位移
        v_grid = grid + flow.permute(0, 2, 3, 1) / norm

        # 4. 采样
        return F.grid_sample(input, v_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
