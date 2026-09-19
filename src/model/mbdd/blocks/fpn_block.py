import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels=256,**kwargs):
        """
        in_channels_list: 输入的各层特征通道数
        out_channels: 融合后的统一通道数
        """
        super(FPNBlock, self).__init__()
        
        # 侧向连接 (Lateral Layers)：统一通道数
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) 
            for in_ch in in_channels_list
        ])


        self.refine_blocks = nn.ModuleList()
        for i in range(len(in_channels_list)):
            curr_expansion = 4
            if i < 2:
                # --- 方案：浅层轻量化（去掉 CBAM，改用简单卷积） ---
                self.refine_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.GELU()
                    )
                )
            else:
                # --- 方案：深层精炼（保留 CBAM） ---
                if kwargs.get('kernel_size', 3) == 3:
                    curr_expansion = 2  # 维持你之前的 2 倍压缩逻辑
                
                self.refine_blocks.append(
                    ConvnextBlock_CBAM(
                        out_channels, 
                        out_channels // curr_expansion, 
                        expansion=curr_expansion,
                        **kwargs
                    )
                )
        
        # 平滑层：消除上采样的混叠效应
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x_list):
        """
        x_list: 包含 [f1, f2, f3, f4] 的列表
        """
        # 1. 先进行侧向变换
        l1, l2, l3, l4 = [lat(f) for lat, f in zip(self.lateral_layers, x_list)]
        
        # 2. 自顶向下融合 (Top-down pathway)
        p4 = self.refine_blocks[3](l4)
        p3_feat = F.interpolate(p4, size=l3.shape[2:], mode='bilinear', align_corners=True) + l3
        p3 = self.refine_blocks[2](p3_feat)
        p2_feat = F.interpolate(p3, size=l2.shape[2:], mode='bilinear', align_corners=True) + l2
        p2 = self.refine_blocks[1](p2_feat)
        p1_feat = F.interpolate(p2, size=l1.shape[2:], mode='bilinear', align_corners=True) + l1
        p1 = self.refine_blocks[0](p1_feat)
        
        out = self.smooth(p1)
        return out

class LightRefineBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch), 
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=1), # PW (1x1)
            nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        return x + self.conv(x)

class ChannelAttentionModule(nn.Module): 
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1) 

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False), 
            nn.GELU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False) 
        )
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x)) 
        # print(avgout.shape) 
        maxout = self.shared_MLP(self.max_pool(x)) 
        return self.sigmoid(avgout + maxout) 


class SpatialAttentionModule(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
        avgout = torch.mean(x, dim=1, keepdim=True) 
        maxout, _ = torch.max(x, dim=1, keepdim=True) 
        out = torch.cat([avgout, maxout], dim=1) 
        out = self.sigmoid(self.conv2d(out)) 
        return out


class CBAM(nn.Module):
    def __init__(self, channel, **kwargs):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel) 
        self.spatial_attention = SpatialAttentionModule(**kwargs) 
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        out = self.channel_attention(x) * x 
        out = self.spatial_attention(out) * out 
        return out


class ConvnextBlock_CBAM(nn.Module):

    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4, **kwargs):
        super(ConvnextBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential( 
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False), 
            nn.BatchNorm2d(places),
            nn.GELU(),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False), 
            nn.BatchNorm2d(places),
            nn.GELU(),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion,**kwargs) 
        self.grn = GRN(places * self.expansion)

        if self.downsampling: 
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.bottleneck(x) 
        out = self.cbam(out) 
        out = self.grn(out)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual 
        out = self.gelu(out) 
        return out

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)# BC HW
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x