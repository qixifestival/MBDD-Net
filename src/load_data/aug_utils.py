import random
import torch.nn as nn
import numpy as np
import albumentations as A
import cv2
import os
from glob import glob

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)
    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    return normalized_img
def normalize_sar(img):
    """针对单通道 SAR 影像的归一化"""
    img_array = np.asarray(img, dtype=np.float32)
    # 1. 对数变换：压缩动态范围（可选，但强烈建议用于 TIF 强度图）
    # 这样可以防止极个别超强散射点（如金属屋顶）导致整体特征变暗
    img_array = np.log1p(np.abs(img_array))
    # 2. Min-Max 归一化到 [0, 1]
    vmin, vmax = np.percentile(img_array, [1, 99.5])
    normalized_img = (img_array - vmin) / (vmax - vmin + 1e-6)
    return np.clip(normalized_img, 0.0, 1.0)

class AugUtils(nn.Module):
    def __init__(self, cloud_path=None):
        super(AugUtils, self).__init__()
        self.cloud_aug=None
        if cloud_path is not None:
            self.cloud_aug=CloudAugmentor(cloud_dir=cloud_path)
        self.geom_aug =A.Compose([
            A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'image2':'image','image3':'image'})
        
        self.sar_aug =A.Compose([
            A.OneOf([
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.5),
            ], p=0.3)
        ], additional_targets={})
        
        self.opt_aug =A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5)
            ],p=0.5)
        ], additional_targets={})

    def set_random_seed(self, seed):
        self.geom_aug.set_random_seed(seed)
        self.sar_aug.set_random_seed(seed + 1)
        self.opt_aug.set_random_seed(seed + 2)

    def forward(self,pre_opt,post_opt,post_sar,label):
        post_sar_3ch = np.concatenate([post_sar]*3, axis=-1)
        geom_out=self.geom_aug(image=pre_opt,image2=post_opt,image3=post_sar_3ch,mask=label)
        pre, post, sar, mask = geom_out['image'], geom_out['image2'], geom_out['image3'], geom_out['mask']
        pre = self.opt_aug(image=pre)['image']
        post = self.opt_aug(image=post)['image']
        sar = self.sar_aug(image=sar)['image']
        sar = sar[:, :, 0:1]
        if self.cloud_aug is not None and random.random()>0.7:
            post=self.cloud_aug.apply_cloud(post,global_alpha=random.uniform(0.7,1.0))
        pre = np.ascontiguousarray(pre)
        post = np.ascontiguousarray(post)
        sar = np.ascontiguousarray(sar)
        mask = np.ascontiguousarray(mask)
        return pre, post, sar, mask

class CloudAugmentor:
    def __init__(self, cloud_dir, preload=True):
        """
        cloud_dir: 存放多张厚积云素材的文件夹路径
        preload: 是否预加载到内存
        """
        print("Load cloud from "+cloud_dir )
        self.cloud_files = sorted(glob(os.path.join(cloud_dir, "*.png")))
        
        if not self.cloud_files:
            raise ValueError(f"在 {cloud_dir} 中没找到任何云素材！")
        
        self.preload = preload
        self.cloud_cache = []

        if self.preload:
            for path in self.cloud_files:
                processed_cloud = self._load_and_process(path)
                if processed_cloud is not None:
                    self.cloud_cache.append(processed_cloud)

    def _load_and_process(self, path):
        """将磁盘上的黑底图转为内存中的 BGRA 透明格式"""
        cloud_bgr = cv2.imread(path)
        if cloud_bgr is None: return None
        
        # 提取灰度作为 Alpha
        gray = cv2.cvtColor(cloud_bgr, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(cloud_bgr)
        # 合并为 4 通道 (H, W, 4)
        return cv2.merge([b, g, r, gray])

    def apply_cloud(self, img_np, global_alpha=0.6, rand=random):
        """
        从缓存中随机挑一张云贴上去
        img_np: 输入的 BGR 图像 (H, W, 3)
        global_alpha: 混合透明度
        """
        h, w = img_np.shape[:2]

        # 1. 随机选择一张预加载好的云
        if self.preload and self.cloud_cache:
            cloud_bgra = rand.choice(self.cloud_cache)
        else:
            return img_np

        cloud_h, cloud_w = cloud_bgra.shape[:2]

        # 2. 随机裁剪 (确保散乱且厚实)
        # 既然是厚云，裁剪比例可以设在 0.3-0.6，这样云块比较集中且硬
        crop_h = int(cloud_h * rand.uniform(0.3, 0.6))
        crop_w = int(cloud_w * rand.uniform(0.3, 0.6))
        start_y = rand.randint(0, cloud_h - crop_h)
        start_x = rand.randint(0, cloud_w - crop_w)
        patch = cloud_bgra[start_y:start_y+crop_h, start_x:start_x+crop_w]

        # 3. 缩放并准备混合
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 提取颜色和 Alpha
        cloud_rgb = patch[:, :, :3].astype(np.float32)
        # 归一化 Alpha 并乘以全局透明度
        cloud_alpha = (patch[:, :, 3].astype(np.float32) / 255.0) * global_alpha
        cloud_alpha = cloud_alpha[:, :, np.newaxis] # 变为 (H, W, 1)

        # 4. 执行 Alpha Blending
        img_float = img_np.astype(np.float32)
        out = (cloud_rgb * cloud_alpha) + (img_float * (1.0 - cloud_alpha))
        
        return np.clip(out, 0, 255).astype(np.uint8)
