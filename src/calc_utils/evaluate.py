import torch

class ConfusionMatrix(object):
    def __init__(self, num_class, ignore_index=None, device='cuda'):
        self.num_class = num_class
        self.device = device
        # 使用 torch.long 对应原代码的 longlong
        self.confusion_matrix = torch.zeros((self.num_class, self.num_class), dtype=torch.long, device=self.device)
        self._epsilon = 1e-7
        self.ignore_index = ignore_index

    def Pixel_Accuracy(self):
        # Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        sum_tp = torch.diag(self.confusion_matrix).sum().float()
        total = self.confusion_matrix.sum().float()
        Acc = sum_tp / (total + self._epsilon)
        return Acc

    def Pixel_Accuracy_Class(self):
        # Acc = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1) + self._epsilon)
        tps = torch.diag(self.confusion_matrix).float()
        res = self.confusion_matrix.sum(dim=1).float()
        Acc = tps / (res + self._epsilon)
        # nanmean 对应 torch.nanmean
        mAcc = torch.mean(Acc) # 如果没有 NaN 的话，如果有 NaN 可以用 Acc[Acc == Acc].mean()
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        # Pre = TP / (FP + TP)
        Pre = self.confusion_matrix[1, 1].float() / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1] + self._epsilon)
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        # Rec = TP / (FN + TP)
        Rec = self.confusion_matrix[1, 1].float() / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1] + self._epsilon)
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre + self._epsilon)
        return F1

    def calculate_per_class_metrics(self):
        # 排除第0类（通常是背景）
        TPs = torch.diag(self.confusion_matrix)[1:].float()
        FNs = self.confusion_matrix.sum(dim=1)[1:].float() - TPs
        FPs = self.confusion_matrix.sum(dim=0)[1:].float() - TPs
        return TPs, FNs, FPs

    def Damage_F1_score(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + self._epsilon)
        recalls = TPs / (TPs + FNs + self._epsilon)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + self._epsilon)
        return f1_scores

    def Mean_Intersection_over_Union(self):
        IoUs = self.Intersection_over_Union()
        MIoU = torch.mean(IoUs)
        return MIoU

    def Intersection_over_Union(self):
        tps = torch.diag(self.confusion_matrix).float()
        union = (torch.sum(self.confusion_matrix, dim=1) + 
                 torch.sum(self.confusion_matrix, dim=0) - 
                 tps).float()
        IoU = tps / (union + self._epsilon)
        return IoU

    def Kappa_coefficient(self):
        num_total = self.confusion_matrix.sum().float()
        observed_accuracy = torch.trace(self.confusion_matrix).float() / (num_total + self._epsilon)
        
        sum_row = self.confusion_matrix.sum(dim=1).float()
        sum_col = self.confusion_matrix.sum(dim=0).float()
        expected_accuracy = torch.sum(sum_row * sum_col) / (num_total * num_total + self._epsilon)

        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy + self._epsilon)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        total = self.confusion_matrix.sum().float()
        freq = self.confusion_matrix.sum(dim=1).float() / (total + self._epsilon)
        iu = self.Intersection_over_Union()

        # 对应 freq[freq > 0]
        mask = freq > 0
        FWIoU = (freq[mask] * iu[mask]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # 确保输入是 torch.Tensor
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].long() + pre_image[mask].long()
        # np.bincount 对应 torch.bincount
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # 统一设备
        gt_image = gt_image.to(self.device)
        pre_image = pre_image.to(self.device)
        
        assert gt_image.shape == pre_image.shape
        if self.ignore_index is None:
            self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        else:
            mask = (gt_image != self.ignore_index)
            if not mask.any():
                return
            gt_image_masked = gt_image[mask]
            pre_image_masked = pre_image[mask]
            self.confusion_matrix += self._generate_matrix(gt_image_masked, pre_image_masked)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class, self.num_class), 
                                            dtype=torch.long, device=self.device)