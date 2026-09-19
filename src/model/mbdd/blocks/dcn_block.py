import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # 1. 学习偏移量的层
        # 保证 offset_conv 的stride和dcn一致， 进而保证offset矩阵大小和卷积窗口滑动的次数匹配
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=True
        )
        
        # 初始化
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # 2. DCN 层
        self.dcn = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )

        # 3. 掩码层：调制变形卷积，忽略对建筑物损毁没有贡献的噪点 DCN v2
        self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, 
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def forward(self, x):
        mask = torch.sigmoid(self.mask_conv(x)) # 每个点都是0.5，初始保持中立
        offset = self.offset_conv(x)# [B, 2*K*K, H_out, W_out]
        return self.dcn(x, offset,mask=mask)