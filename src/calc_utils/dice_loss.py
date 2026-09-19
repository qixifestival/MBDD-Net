import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """
    函数版 Dice Loss，支持 ignore_index=255
    pred: [B, 1, H, W] 原始输出(logits)
    target: [B, H, W] 包含 0, 1, 255 的标签
    """
    # 1. 激活与屏蔽掩码
    # 使用 sigmoid 将 logits 转为概率
    pred_probs = torch.sigmoid(pred)
    
    # 建立有效区域掩码 (屏蔽半损毁 255)
    mask = (target != 255).float().unsqueeze(1)
    
    # 只把类别 1 (全损毁) 作为正样本
    target_f = (target == 1).float().unsqueeze(1)
    
    # 应用掩码：将无效区域的预测和标签全部清零
    # 这样它们既不贡献交集(intersection)，也不贡献并集(union)
    valid_pred = pred_probs * mask
    valid_target = target_f * mask

    # 2. 计算 Dice (在空间维度 H, W 上求和)
    # dim=(1, 2, 3) 确保在 Channel, Height, Width 上求和，保留 Batch 维度
    intersection = (valid_pred * valid_target).sum(dim=(1, 2, 3))
    union = valid_pred.sum(dim=(1, 2, 3)) + valid_target.sum(dim=(1, 2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # 返回均值损失
    return 1 - dice.mean()
