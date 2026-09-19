from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18
from .dcn_block import DeformableConvBlock

class SarEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.pretrained_param_ids = set()
        self.new_param_ids = set()

        base_model = self._load_s1_backbone(checkpoint_path)
        self._convert_resnet_full(base_model)
        self.model = base_model

    def _load_s1_backbone(self, checkpoint_path):
        # 实例化标准 ResNet18
        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        loaded_pretrained = checkpoint_path is not None
        if loaded_pretrained:
            checkpoint_path = Path(checkpoint_path).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"SAR pretrained checkpoint not found: {checkpoint_path}. "
                    "See docs/PRETRAINED_WEIGHTS.md."
                )
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            clean_sd = {k: v for k, v in state_dict.items() if 'fc.' not in k}
            backbone.load_state_dict(clean_sd, strict=False)
        else:
            print("Warning: SAR encoder is randomly initialized. "
                  "Pass --sar_pretrained_path to reproduce the paper protocol.")
        backbone.fc = torch.nn.Identity()

        for p in backbone.parameters():
            target = self.pretrained_param_ids if loaded_pretrained else self.new_param_ids
            target.add(id(p))
        return backbone
    
    def _recursive_update(self,module, name_str=""):
        for name, child in module.named_children():
            full_name = f"{name_str}.{name}" if name_str else name
            
            # 情况 A: 替换 ReLU -> GELU (对所有模块生效)
            if isinstance(child, (nn.ReLU, nn.ReLU6)):
                setattr(module, name, nn.GELU())
            
            # 情况 B: 针对 layer 3, 4 的 BasicBlock 替换 DCN
            elif any(x in full_name for x in ['layer2','layer3', 'layer4']) and \
                 child.__class__.__name__ == 'BasicBlock':
                
                # 替换 conv1 和 conv2 为 DCN
                for conv_name in ['conv1', 'conv2']:
                    old_conv = getattr(child, conv_name)
                    old_w = old_conv.weight.data.clone()

                    for p in old_conv.parameters():
                        self.pretrained_param_ids.discard(id(p))
                    
                    new_dcn = DeformableConvBlock(
                        old_conv.in_channels, old_conv.out_channels,
                        stride=old_conv.stride, padding=old_conv.padding
                    )
                    
                    with torch.no_grad():
                        new_dcn.dcn.weight.copy_(old_w)
                    
                    setattr(child, conv_name, new_dcn)

                    for p in new_dcn.parameters():
                        self.new_param_ids.add(id(p))
        
                self._recursive_update(child, full_name)
            
            # 情况 C: 其他容器模块继续向下递归
            else:
                self._recursive_update(child, full_name)
    
    def _convert_resnet_full(self, model):
        # --- 1. 重构Stem ---
        old_stem_weight = model.conv1.weight.data.clone()
        for p in model.conv1.parameters():
            self.pretrained_param_ids.discard(id(p))
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model.gelu = nn.GELU()
        model.relu = nn.GELU()
        
        with torch.no_grad():
            if old_stem_weight.shape[1] == 2:
                fused_weight=old_stem_weight.mean(dim=1, keepdim=True)
                print("📡 Detection: S1 Pretrained (2ch) -> fused to 1ch")
            elif old_stem_weight.shape[1] == 3:
                fused_weight=old_stem_weight.mean(dim=1, keepdim=True)
                print("📸 Detection: Optical Pretrained (3ch) -> fused to 1ch")
            else:
                fused_weight=old_stem_weight

            model.conv1.weight.copy_(fused_weight)

        for p in model.conv1.parameters():
            self.new_param_ids.add(id(p))

        # 执行递归
        self._recursive_update(model)
        
    def forward(self, x):
        model=self.model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.gelu(x) 
        x = model.maxpool(x)
        f1 = model.layer1(x)
        f2 = model.layer2(f1)
        f3 = model.layer3(f2)
        f4 = model.layer4(f3)
        return [f1, f2, f3, f4]
    
class ResNetDStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # 3层 3x3 堆叠，第一层 stride=2 进行下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.stem(x)
