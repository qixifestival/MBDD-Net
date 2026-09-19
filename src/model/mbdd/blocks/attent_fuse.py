import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple learnable scale parameter (replacement for mmcv's Scale)
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

# ConvModule replacement
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, norm_cfg=True, act_cfg=True):
        super(ConvModule, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)]
        if norm_cfg:
            layers.append(nn.BatchNorm2d(out_channels))
        if act_cfg:
            layers.append(nn.GELU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# SpatialQKVBlock
class SpatialQKVBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(SpatialQKVBlock, self).__init__()
        self.conv_q = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_k = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_v = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)

    def forward(self, x):
        N, C, H, W = x.shape
        x_q = self.conv_q(x).reshape(N, -1, H * W).permute(0, 2, 1).contiguous()
        x_k = self.conv_k(x).reshape(N, -1, H * W)
        x_v = self.conv_v(x).reshape(N, -1, H * W).permute(0, 2, 1).contiguous()
        return x_q, x_k, x_v

# SpatialAttBlock
class SpatialAttBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(SpatialAttBlock, self).__init__()
        self.qkv_opt = SpatialQKVBlock(in_channels, channels)
        self.qkv_sar = SpatialQKVBlock(in_channels, channels)
        self.gamma_opt = Scale(0)
        self.gamma_sar = Scale(0)
        self.project_opt = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)
        self.project_sar = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)

    def forward(self, x_opt, x_sar):
        N, C, H, W = x_opt.shape
        q_opt, k_opt, v_opt = self.qkv_opt(x_opt)
        q_sar, k_sar, v_sar = self.qkv_sar(x_sar)

        att_opt = F.softmax(torch.bmm(q_opt, k_opt), dim=-1)
        att_sar = F.softmax(torch.bmm(q_sar, k_sar), dim=-1)
        att = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att, v_opt).permute(0, 2, 1).contiguous().reshape(N, -1, H, W)
        s_opt = self.project_opt(s_opt)
        s_opt = x_opt + self.gamma_opt(s_opt)

        s_sar = torch.bmm(att, v_sar).permute(0, 2, 1).contiguous().reshape(N, -1, H, W)
        s_sar = self.project_sar(s_sar)
        s_sar = x_sar + self.gamma_sar(s_sar)

        return s_opt + s_sar

# ChannelQKVBlock
class ChannelQKVBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(ChannelQKVBlock, self).__init__()
        self.conv_q = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_k = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_v = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)

    def forward(self, x):
        N, C, H, W = x.shape
        x_q = self.conv_q(x).reshape(N, -1, H * W)
        x_k = self.conv_k(x).reshape(N, -1, H * W).permute(0, 2, 1).contiguous()
        x_v = self.conv_v(x).reshape(N, -1, H * W)
        return x_q, x_k, x_v

# ChannelAttBlock
class ChannelAttBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(ChannelAttBlock, self).__init__()
        self.qkv_opt = ChannelQKVBlock(in_channels, channels)
        self.qkv_sar = ChannelQKVBlock(in_channels, channels)
        self.gamma_opt = Scale(0)
        self.gamma_sar = Scale(0)
        self.project_opt = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)
        self.project_sar = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)

    def forward(self, x_opt, x_sar):
        N, C, H, W = x_opt.shape
        q_opt, k_opt, v_opt = self.qkv_opt(x_opt)
        q_sar, k_sar, v_sar = self.qkv_sar(x_sar)

        att_opt = F.softmax(torch.bmm(q_opt, k_opt), dim=-1)
        att_sar = F.softmax(torch.bmm(q_sar, k_sar), dim=-1)
        att = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att, v_opt).reshape(N, -1, H, W)
        s_opt = self.project_opt(s_opt)
        s_opt = x_opt + self.gamma_opt(s_opt)

        s_sar = torch.bmm(att, v_sar).reshape(N, -1, H, W)
        s_sar = self.project_sar(s_sar)
        s_sar = x_sar + self.gamma_sar(s_sar)

        return s_opt + s_sar

class AttFuseBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(AttFuseBlock, self).__init__()
        self.sab = SpatialAttBlock(in_channels, channels, out_channels)
        self.cab = ChannelAttBlock(in_channels, channels, out_channels)

    def forward(self, x_opt, x_sar):
        sab_feat = self.sab(x_opt, x_sar)
        cab_feat = self.cab(x_opt, x_sar)
        return sab_feat + cab_feat
