
import torch
import torch.nn as nn
import timm

class OptEncoder(nn.Module):
    def __init__(self, model_name='convnext_tiny.fb_in22k_ft_in1k_384', pretrained=True):
        super().__init__()
        # 使用 features_only=True 主要是为了获取各层通道数等信息
        base_model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.stem = nn.Sequential(
            base_model.stem_0,
            base_model.stem_1
        )
        
        # 2. 提取共享的 Stage 0 和 1
        self.shared_stage0 = base_model.stages_0
        self.shared_stage1 = base_model.stages_1
        
        # 3. 提取定位分支 (Localization)
        self.loc_stage2 = base_model.stages_2
        self.loc_stage3 = base_model.stages_3
        
        # 4. 提取损毁分类分支 (Damage) 并深拷贝
        import copy
        self.dam_stage2 = copy.deepcopy(base_model.stages_2)
        self.dam_stage3 = copy.deepcopy(base_model.stages_3)
        
        
    def forward(self, x):
        # 提取共享特征
        x_stem = self.stem(x)
        x1 = self.shared_stage0(x_stem)
        x2 = self.shared_stage1(x1)
        
        # 运行定位分支 (通常用于处理 Pre-disaster 图像提取边缘和建筑物位置)
        loc3 = self.loc_stage2(x2)
        loc4 = self.loc_stage3(loc3)
        
        # 运行损毁分支 (用于提取灾后或灾前的状态特征)
        dam3 = self.dam_stage2(x2)
        dam4 = self.dam_stage3(dam3)
        
        # 返回：[定位特征组, 损毁分类特征组]
        # 注意：x1, x2 是共享的，loc3/4 和 dam3/4 是任务特定的
        return [x1, x2, loc3, loc4], [x1, x2, dam3, dam4]