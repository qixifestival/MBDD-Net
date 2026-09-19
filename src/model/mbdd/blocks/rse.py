import torch
import torch.nn as nn
import torch.nn.functional as F

class RES_Dilated_Conv(nn.Module):
    """
    Wide-Focus Residual Block with dilated convolutions.
    """

    def __init__(self, in_channels, out_channels):
        super(RES_Dilated_Conv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False) #dilation=1
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=2, bias=False) #dilation=2
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=3, bias=False) #dilation=3
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False) #dilation=1
        
        # Ensure input/output have same number of channels for residual connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
        # Optional: BatchNorm for better training stability
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # First dilated conv branch
        x1 = self.conv1(x)
        x1 = self.bn1(x1) 
        x1 = F.gelu(x1) 
        x1 = F.dropout(x1, p=0.1) 

        # Second dilated conv branch
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, p=0.1)

        # Third dilated conv branch
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, p=0.1)

        # Combine the branches
        added = torch.add(x1, x2)
        added = torch.add(added, x3)

        # Apply final convolution
        x_out = self.conv4(added)
        x_out = self.bn4(x_out)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, p=0.1)

        # Add residual connection (shortcut)
        residual = self.shortcut(x)  # Either identity or 1x1 conv
        x_out += residual  # Residual connection
        return x_out
