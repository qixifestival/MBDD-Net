import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import timm
from timm.layers import DropPath
from torchvision.models import resnet50
from .blocks.flow_resnet import FlowN_ResNet as Flow
from .blocks.cross_modal_attention import CrossModalFusion
from .blocks.fpn_block import FPNBlock
from .blocks.aspp import ASPPHead
from .blocks.buildpred import BuildingHead
from .blocks.rse import RES_Dilated_Conv
from .blocks.sar_encoder import SarEncoder

class MBDD(nn.Module):
    def __init__(self, num_classes, use_fam=True, sar_pretrained_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.use_fam = use_fam
        # 1. Encoder
        self.opt_encoder = timm.create_model('convnext_tiny.fb_in22k_ft_in1k_384', pretrained=True, features_only=True)
        self.sar_encoder = SarEncoder(checkpoint_path=sar_pretrained_path)
        
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.opt_dim = [96, 192, 384, 768]
        self.sar_dim = [64, 128, 256, 512]

        # 3. Building Location Decoder
        self.fpn_build = FPNBlock(self.opt_dim, out_channels=256, kernel_size=3)# use small kernel to protect building boundary
        self.building_head = BuildingHead(in_ch=256, edge_ch=64, num_classes=1)

        # 4. Building Damage Detection Decoder
        # 4.1 Zip the opt features (dilated conv)  
        self.rse_opt= nn.ModuleList([RES_Dilated_Conv(in_channels=self.opt_dim[i], out_channels=self.sar_dim[i]) for i in range(len(self.sar_dim))])
        # 4.2 Align pre-post imgs
        self.flow_align_layers = nn.ModuleList(
            [Flow(self.sar_dim[0]), Flow(self.sar_dim[1])] if self.use_fam else []
        )
        self.criterion = nn.SmoothL1Loss() # smooth align loss
        # 4.3 Cross-Modal Fusion
        self.norm_opt = nn.ModuleList([nn.LayerNorm(d) for d in self.sar_dim])
        self.norm_sar = nn.ModuleList([nn.GroupNorm(min(d // 4,32), d) for d in self.sar_dim])
        self.fusion = CrossModalFusion(self.sar_dim)
        # 4.4 Head of prediction
        self.fpn_damage = FPNBlock(self.sar_dim, out_channels=256,kernel_size=7) # use bigger kernel to get the damage area
        self.damage_spatial_drop = nn.Dropout2d(p=0.3)
        self.damage_head = ASPPHead(in_ch=256,low_channels=64, num_classes=num_classes)
        self.sar_aux_fpn = FPNBlock(self.sar_dim, out_channels=256,kernel_size=7)
        self.sar_aux_spatial_drop = nn.Dropout2d(p=0.3)
        self.sar_aux_head = ASPPHead(in_ch=256, low_channels=64, num_classes=num_classes)

    

    def align(self, pre, post, label=None):
        loss = 0
        flows = []
        new_post = list(post) 

        for i in range(len(self.flow_align_layers)):
            # 1. 得到光流对齐后的特征
            aligned_feat, flow = self.flow_align_layers[i](pre[i], post[i])
            new_post[i] = aligned_feat
            flows.append(flow)

            if label is not None:
                mask = F.interpolate(label.float(), size=post[i].size()[-2:], mode='nearest')

                # 2. 计算相似度权重 (基于 L1 距离)
                with torch.no_grad():
                    diff = torch.abs(pre[i] - aligned_feat).mean(dim=1, keepdim=True)
                    sim_weight = torch.exp(-diff) 

                # 3. 结合建筑掩码：只在“是建筑”且“没怎么变”的地方算对齐 Loss
                final_mask = mask * sim_weight
                loss += self.criterion(pre[i] * final_mask, aligned_feat * final_mask)

                # 4. 引入 TV Loss 约束光流平滑
                # 即使建筑没了，光流也要跟着周围的马路位移走，不能乱扭曲
                df_dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                df_dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
                loss += 0.01 * (df_dy.mean() + df_dx.mean())

        if label is not None:
            return new_post, loss / len(self.flow_align_layers)
        return new_post, flows

            
    def forward(self, pre_opt, post_opt, post_sar, label_loc=None,throw_sar_p=0.0,throw_opt_p=0.0):
        align_loss = torch.tensor(0.0).to(pre_opt.device)

        # post_opt=torch.zeros_like(post_opt).detach()

        # --- Stage 1:   Feature Extraction ---
        feats_pre_raw = self.opt_encoder(pre_opt)
        feats_post_raw = self.opt_encoder(post_opt)
        feats_sar=self.sar_encoder(post_sar)
        feat_edge=self.edge_extractor(pre_opt)
        feat_post_edge=self.edge_extractor(post_opt)


        
        # --- Stage 2:   Building Location Prediction ---
        p1_build = self.fpn_build(feats_pre_raw)
        out_build = self.building_head(p1_build, feat_edge)


        # --- Stage 3:   Building Damage Detection ---
        feats_pre = [self.rse_opt[i](feats_pre_raw[i]) for i in range(len(feats_pre_raw))]
        feats_post = [self.rse_opt[i](feats_post_raw[i]) for i in range(len(feats_post_raw))]

        # --- Stage 3.1 :  Calculate aux sar head ---
        if self.training:
            aux_sar_feat=self.sar_aux_fpn(feats_sar)
            aux_sar_feat=self.sar_aux_spatial_drop(aux_sar_feat)
            aux_sar_damage,_ =self.sar_aux_head(aux_sar_feat, feat_post_edge, out_build)

        if self.training and random.random() < throw_opt_p:
            feats_post = [torch.zeros_like(f) for f in feats_post]
        elif self.training and random.random() < throw_sar_p:
            feats_sar = [torch.zeros_like(f) for f in feats_sar]

        if self.use_fam:
            if label_loc is not None and self.training:
                ann=label_loc.clone().unsqueeze(1).detach()
                feats_post, align_loss = self.align(feats_pre, feats_post, ann)
            else:
                feats_post, _ = self.align(feats_pre, feats_post)
        
        for i in range(4):
            # Opt 做 LN (需 permute)
            feats_pre[i] = self.norm_opt[i](feats_pre[i].permute(0,2,3,1)).permute(0,3,1,2)
            feats_post[i] = self.norm_opt[i](feats_post[i].permute(0,2,3,1)).permute(0,3,1,2)

            # SAR 做精细 GN
            feats_sar[i] = self.norm_sar[i](feats_sar[i])

        fused_pyramid = self.fusion(feats_pre, feats_post, feats_sar)

        p1_damage = self.fpn_damage(fused_pyramid)
        p1_damage = self.damage_spatial_drop(p1_damage)
        if self.training:
            out_damage, aux_damage = self.damage_head(p1_damage, feat_post_edge, out_build)
        else:
            out_damage = self.damage_head(p1_damage, feat_post_edge,out_build)
            
        out_build = F.interpolate(out_build, scale_factor=2, mode='bilinear', align_corners=True)
        out_damage = F.interpolate(out_damage, scale_factor=2, mode='bilinear', align_corners=True)

        if self.training:
            return out_build, out_damage, align_loss, aux_damage, aux_sar_damage
        
        return out_build, out_damage,None,None,None

def get_optim(model, weight_decay=0.03, learning_rate=1e-4):
    params_group = []
    
    # 提取光学分支 ID (假设光学没做手术)
    opt_ids = [id(p) for p in model.opt_encoder.parameters()]

    for name, param in model.named_parameters():
        p_id = id(param)
        
        # 1. 光学分支
        if p_id in opt_ids:
            params_group.append({'params': param, 'lr': learning_rate * 0.5, 'weight_decay': 0.1})
            
        # 2. SAR 分支 - 预训练老层
        elif p_id in model.sar_encoder.pretrained_param_ids:
            params_group.append({'params': param, 'lr': learning_rate * 0.3, 'weight_decay': 0.1})
            
        # 3. SAR 分支 - 重构新层 (Stem / DCN)
        elif p_id in model.sar_encoder.new_param_ids:
            params_group.append({'params': param, 'lr': learning_rate, 'weight_decay': weight_decay})
            
        # 4. 其他 (Decoder, Head)
        else:
            params_group.append({'params': param, 'lr': learning_rate, 'weight_decay': weight_decay})

    optimizer=torch.optim.AdamW(params_group)

    for group in optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = group['lr']

    return optimizer
