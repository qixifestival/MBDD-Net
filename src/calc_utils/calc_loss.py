import torch
import torch.nn as nn
import torch.nn.functional as F
import calc_utils.lovasz_loss as L
import calc_utils.dice_loss as D
from torchvision.ops import sigmoid_focal_loss



class MBDDLoss(nn.Module):
    def __init__(self, max_iters=31250, writer=None):
        super(MBDDLoss, self).__init__()
        self.max_iters = max_iters
        self.writer = writer
        self.current_step = 0
        self.last_phase = -1
        self.last_aux_phase = -1

    def update_step(self, step=None):
        if step is not None: self.current_step = step
        else: self.current_step += 1

    def forward(self, out_loc, out_clf, label_loc, label_clf, aux_clf=None, aux_sar=None, align_loss=0.0, state="train"):
        # 1. 定位损失 (Building Location)
        bce_loc = F.binary_cross_entropy_with_logits(out_loc.squeeze(1), (label_loc == 1).float(), reduction='none')
        mask_loc = (label_loc != 255).float()
        loss_location = (bce_loc * mask_loc).sum() / (mask_loc.sum() + 1e-6) + D.dice_loss(out_loc, label_loc)

        # 2. 分类损失 (Damage Classification)
        loss_classification = self.calc_clf_loss(out_clf, label_clf,w1=0.6,w2=0.4)

        # 3. 辅助头损失 (Aux Damage)
        loss_aux_total = torch.tensor(0.0).to(out_clf.device)
        loss_sar_aux = torch.tensor(0.0).to(out_clf.device)
        w_aux,w_sar = self._get_dynamic_aux_weight()
        
        if aux_clf is not None and w_aux > 0:
            aux_clf_up = F.interpolate(aux_clf, size=label_clf.shape[-2:], mode='bilinear', align_corners=True)
            aux_target_masked = label_clf.clone()
            aux_target_masked[label_loc == 0] = 255 
            loss_aux_total = self.calc_clf_loss(aux_clf_up, aux_target_masked)
        
        if aux_sar is not None and w_sar > 0:
            aux_sar_up = F.interpolate(aux_sar, size=label_clf.shape[-2:], mode='bilinear', align_corners=True)
            wmap,safe_zone=self.get_weight_map(label_loc,label_clf)
            sar_target_masked = label_clf.clone()
            sar_target_masked[safe_zone==0] = 255
            loss_sar_aux = self.calc_clf_loss(aux_sar_up, sar_target_masked,weight_map=wmap)

        final_loss = loss_location * 0.3 + loss_classification * 4.0 + w_aux * loss_aux_total + align_loss * 0.05 + w_sar* loss_sar_aux
        
        # --- 自动打印 Scalar 到 TensorBoard ---
        if state == "train":
            if self.writer is not None and (self.current_step+1) %10 ==0 :
                step = self.current_step + 1 
                self.writer.add_scalar("Loss/train", final_loss.item(), step)
                self.writer.add_scalar("LossDetail/train_loc", loss_location.item(), step)
                self.writer.add_scalar("Loss/train_clf", loss_classification.item(), step)
                self.writer.add_scalar("LossDetail/align_loss",align_loss,step)
                self.writer.add_scalar("LossDetail/aux_clf",loss_aux_total,step)
                self.writer.add_scalar("LossDetail/aux_sar",loss_sar_aux,step)
            self.update_step() # 每次 forward 自动累加步数

        return final_loss, loss_location, loss_classification
    
    def get_weight_map(self, label_loc, label_clf):

        weights = torch.full_like(label_loc, fill_value=0.01)
        weights[label_clf == 1] = 1.8

        dilated_mask_soft = F.max_pool2d(label_loc.unsqueeze(1).float(), kernel_size=3, stride=1, padding=1).squeeze(1)
        # 膨胀区域内但不是标注损毁的地方，给 0.5 的容错权重
        buffer_zone = (dilated_mask_soft > 0.8) & (label_clf == 0)
        weights[buffer_zone] = 0.2

        safe_zone = F.max_pool2d(label_loc.unsqueeze(1).float(), kernel_size=7, stride=1, padding=3).squeeze(1)

        return weights, safe_zone

    def _get_dynamic_aux_weight(self):
        """
        w_aux: 融合后的辅助头权重 (逐渐退出，让位于最终融合结果)
        w_sar: SAR 专用的辅助头权重 (中前期保持，后期微调，保护专家权重)
        """
        progress = min(self.current_step / self.max_iters, 1.0)

        # --- 阶段 0: 延长保护期 (0% - 85%) ---
        if progress < 0.85:
            w_aux, w_sar, aux_p = 0.2, 0.5, 0 

        # --- 阶段 1: 压缩衰减期 (85% - 95%) ---
        elif progress < 0.95:
            phase_progress = (progress - 0.85) / 0.1
            w_aux = 0.2 * (1.0 - phase_progress)
            w_sar = 0.5 - (phase_progress * 0.3)
            aux_p = 1

        else:
            w_aux = 0.0
            w_sar = 0.2  # 稍微提高保底权重，确保最后精调阶段 SAR 依然有话语权
            aux_p = 2

        # 日志记录 (同步记录两个权重)
        if self.writer is not None and aux_p != self.last_aux_phase:
            msg = (f"### Aux Weight Shift\n"
                   f"* Phase: {aux_p}\n"
                   f"* W_Main_Aux: {w_aux:.4f}\n"
                   f"* W_SAR_Aux: {w_sar:.4f}\n"
                   f"* Progress: {progress:.2%}")
            self.writer.add_text('Strategy/Weight_Shift', msg, self.current_step)
            self.last_aux_phase = aux_p

        return w_aux, w_sar
    
    def smooth_loss(self,out_clf, target, alpha_f=0.7, gamma_f=1.5, smoothing=0.1):
        valid_mask = (target != 255).float()
        target_clean = torch.where(target == 255, torch.zeros_like(target), target).float()
        target_smoothed = target_clean * (1.0 - smoothing) + 0.5 * smoothing

        # 2. 调用原始的 sigmoid_focal_loss
        # 注意：确保 target_smoothed 已经转换为和 out_clf 相同的 dtype 和 device
        loss = sigmoid_focal_loss(
            out_clf.squeeze(1), 
            target_smoothed, 
            alpha=alpha_f, 
            gamma=gamma_f, 
            reduction='none'
        )

        loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        return loss
    def calc_clf_loss(self, logits, labels, weight_map=None,w1=0.7,w2=0.3):
                # 1. 映射到概率空间
        probs = torch.sigmoid(logits)
        if probs.dim() == 4:
            probs = probs.squeeze(1)


        ignore_index=255
        self.eps=1e-7

        # 2. 准备掩码和标签
        # 过滤 ignore_index (255)
        valid_mask = (labels != ignore_index).float()
        y_true = labels.float()
        
        # 3. LWCE 公式实现: LWCE = -(w1 * Y * log(Y_hat) + w2 * (1 - Y) * log(1 - Y_hat))
        # 使用 clamp 限制概率范围，双重保险防止 NaN
        y_pred = torch.clamp(probs, self.eps, 1.0 - self.eps)

        term_positive = w1 * y_true * torch.log(y_pred) 
        term_negative = w2 * (1.0 - y_true) * torch.log(1.0 - y_pred)
        
        # 4. 应用掩码并求和
        pixel_loss = -(term_positive + term_negative) * valid_mask

        if weight_map is not None:
            if weight_map.dim() == 4: 
                weight_map = weight_map.squeeze(1)
            weight_map = weight_map.to(pixel_loss.device)
            pixel_loss = pixel_loss * weight_map
    
        final_loss = (pixel_loss * valid_mask).sum() / (valid_mask.sum() + self.eps)
        
        return final_loss
