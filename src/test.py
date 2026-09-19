import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from load_data.make_data_loader import MultimodalDamageAssessmentDatset_Inference 
from model.mbdd.MBDD import MBDD
from calc_utils.evaluate import ConfusionMatrix

import ttach as tta

class ModelWrapper(nn.Module):
    def __init__(self, model_name, model_path=None, disable_fam=False, sar_pretrained_path=None):
        super(ModelWrapper, self).__init__()
        self.model_name=model_name
        if self.model_name=='MBDD':
            self.deep_model = MBDD(
                num_classes=1,
                use_fam=not disable_fam,
                sar_pretrained_path=sar_pretrained_path,
            )
        if model_path is not None: 
            self.load_checkpoint(self.deep_model,model_path)

    def load_checkpoint(self, model, path):
        checkpoint = torch.load(path, map_location='cpu')
        
        # 1. 兼容字典格式
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 2. 清洗前缀和 EMA 特有 Key
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == 'n_averaged': continue # 剔除 EMA 计数器
            name = k.replace('module.', '') # 统一去掉包装前缀
            new_state_dict[name] = v
            
        # 3. 柔性加载
        msg = model.load_state_dict(new_state_dict, strict=False)

    def forward(self, combined_tensor):
        if self.model_name=='MBDD':
            pre = combined_tensor[:, 0:3, :, :]
            post = combined_tensor[:, 3:6, :, :]
            sar =combined_tensor[:, 6:7, :, :]
            output_loc, output_clf, _,_,_ = self.deep_model(pre, post,sar)
            combined_output = torch.cat([output_loc, output_clf], dim=1)
            return combined_output
class Trainer(object):

    # 初始化
    def __init__(self, args):

        self.args = args

        # Define the evaluator about the classification task and location task
        self.evaluator_loc = ConfusionMatrix(num_class=2,ignore_index=255)
        self.evaluator_clf = ConfusionMatrix(num_class=2,ignore_index=255)
        self.single_evaluator = ConfusionMatrix(num_class=2,ignore_index=255)

        self.deep_model = ModelWrapper(
            self.args.model_name,
            self.args.model_path,
            disable_fam=self.args.disable_fam,
            sar_pretrained_path=self.args.sar_pretrained_path,
        )
        self.deep_model = self.deep_model.cuda() 

        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ])
        self.tta_model = tta.SegmentationTTAWrapper(self.deep_model, transforms)
        self.inference_model = self.deep_model if self.args.disable_tta else self.tta_model

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
    
    def sliding_window_inference(self, input, window_size=256,stride=128,channels=1,num_classes=1):
        """
        Args:
            model: 训练好的模型
            image: 输入图,Tensor格式 (1, C, H, W)
            window_size: 窗口尺寸 (例如 512)
            stride: 步长 (例如 256, 建议 stride < window_size 以产生重叠)
            num_classes: 分割类别数
        """
        
        b, c, h, w = input.shape

        # 1. 初始化全黑的预测画布和计数器（用于记录每个像素被叠加了多少次）
        full_probs = torch.zeros((b, channels, h, w)).to(input.device)
        count_mask = torch.zeros((b, 1, h, w)).to(input.device)
        y_range = range(0, h - window_size + 1, stride)
        x_range = range(0, w - window_size + 1, stride)
        from itertools import product
        grid = list(product(y_range, x_range))

        # 2. 开始滑动
        with torch.no_grad():
            for y, x in tqdm(grid, desc="Processing Windows", leave=False):
                # 裁剪 Patch
                patch = input[:, :, y:y+window_size, x:x+window_size]
                # 模型推理
                output_probs = self.inference_model(patch) # (1, num_classes, 512, 512)
                # 转为 Softmax 概率
                if num_classes <= 1:
                    probs = torch.sigmoid(output_probs)
                else:
                    probs = F.softmax(output_probs, dim=1)
                # 将结果累加到画布上
                full_probs[:, :, y:y+window_size, x:x+window_size] += probs
                count_mask[:, :, y:y+window_size, x:x+window_size] += 1

        # 3. 处理边缘无法被步长整除的剩余区域 (可选，如果图像是1024而窗口是512则无需此步)
        # 若图像尺寸不固定，建议先对图像进行 Padding 到 stride 的倍数

        # 4. 均值化（消除重叠区域因累加导致的数值过大）
        full_probs /= count_mask

        return full_probs
        
    def model_forward(self, pre_change_imgs, post_change_imgs,post_change_sar):
        output_loc_binary=None
        if self.args.model_name=='MBDD':
            tta_input = torch.cat([pre_change_imgs, post_change_imgs, post_change_sar], dim=1)
            if self.args.use_slide:
                combined_tta_out = self.sliding_window_inference(tta_input,num_classes=1,channels=2)
            else:
                combined_tta_out = self.inference_model(tta_input)
                combined_tta_out=torch.sigmoid(combined_tta_out)

            output_loc = combined_tta_out[:, 0:1, :, :]
            output_clf = combined_tta_out[:, 1:2, :, :]
            output_loc_binary = (output_loc.detach().float() > 0.5).byte().squeeze(1)
            output_clf_binary = (output_clf.detach().float() > 0.5).byte().squeeze(1)

        return  output_loc_binary, output_clf_binary

    def save_original_map(self, prediction, file_name):
        """Saves the colored damage map."""
        # color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        # for cls, color in self.color_map.items():
        #     color_map_img[prediction == cls] = color
        save_img = (prediction.clip(0, 1) * 255)
        output_path = os.path.join(self.args.output_dir, file_name + '_building_damage.png')
        Image.fromarray(save_img.detach().cpu().numpy().astype(np.uint8)).save(output_path)

    def test(self):
        print('---------starting test-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.deep_model.eval()
        test_dataset = MultimodalDamageAssessmentDatset_Inference(self.args.test_dataset_path, self.args.test_data_name_list, type='test', cloud_path=self.args.cloud_path)
        test_data_loader = DataLoader(test_dataset, batch_size=self.args.test_batch_size, num_workers=self.args.num_workers, drop_last=False, shuffle=False)
        torch.cuda.empty_cache()
        iou_rec={}
        loc_scores={}
        clf_scores={}
        score={}

        with torch.no_grad():
            for _, data in tqdm(enumerate(test_data_loader),total=len(test_data_loader)):
                pre_change_imgs, post_change_imgs, post_change_sar, labels_loc, labels_clf, data_idx = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                post_change_sar = post_change_sar.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                output_loc_binary, output_clf_binary =self.model_forward(pre_change_imgs, post_change_imgs, post_change_sar)

                for i in range(len(data_idx)):
                    noTarget_flag = not torch.any(labels_clf[i] == 1)
                    noPredict_flag = not torch.any(output_clf_binary[i] == 1)
                    if noTarget_flag and noPredict_flag:
                        continue
                    if self.args.save_map:
                        # self.save_original_map(output_loc_binary[i], "LOC_"+data_idx[i])
                        self.save_original_map(output_clf_binary[i], "CLF_"+data_idx[i])
                    if self.args.save_per_iou:
                        if noTarget_flag:
                            iou_rec[data_idx[i]] = "False Predict"
                        elif noPredict_flag:
                            iou_rec[data_idx[i]] = "Miss Target"
                        else:
                            self.single_evaluator.reset()
                            self.single_evaluator.add_batch(labels_clf[i], output_clf_binary[i])
                            damage_iou_single = self.single_evaluator.Intersection_over_Union()[1]
                            iou_rec[data_idx[i]] = damage_iou_single.item()

                if output_loc_binary is not None:
                    self.evaluator_loc.add_batch(labels_loc.long(), output_loc_binary.long())
                
                self.evaluator_clf.add_batch(labels_clf.long(), output_clf_binary.long())
        if output_loc_binary is not None:        
            loc_f1_score = self.evaluator_loc.Pixel_F1_score()
            loc_OA = self.evaluator_loc.Pixel_Accuracy()
            loc_mIoU = self.evaluator_loc.Mean_Intersection_over_Union()
            build_iou = self.evaluator_loc.Intersection_over_Union()[1]
            loc_scores={
                'loc f1 (val)': loc_f1_score.item() * 100 if loc_f1_score is not None else 0,
                'loc OA (val)': loc_OA.item() * 100,
                'loc mIoU (val)': loc_mIoU.item() * 100,
                'buildIou (val)': build_iou.item() * 100,
            }
            score["LOC_TASK"]=loc_scores
            print(f'LOC:F1-score is {loc_f1_score.item()}, OA is {loc_OA.item()}, mIoU is {loc_mIoU.item()}, IoU of building is {build_iou.item()}')


        clf_f1_score = self.evaluator_clf.Pixel_F1_score()
        clf_OA = self.evaluator_clf.Pixel_Accuracy()
        damage_iou = self.evaluator_clf.Intersection_over_Union()[1]
        clf_mIoU = self.evaluator_clf.Mean_Intersection_over_Union()
        clf_scores = {
            'clf f1 (val)': clf_f1_score.item() * 100,
            'clf OA (val)': clf_OA.item() * 100,
            'clf mIoU (val)': clf_mIoU.item() * 100,
            'damageIou (val)': damage_iou.item() * 100
        }
        score["CLF_TASK"]=clf_scores
        print(f'CLF:F1-score is {clf_f1_score.item()}, OA is {clf_OA.item()}, mIoU is {clf_mIoU.item()}, IoU of damage is {damage_iou.item()}')

        import json
        score_json_path = os.path.join(self.args.output_dir, 'evaluation_scores.json')
        with open(score_json_path, 'w') as f:
            json.dump(score, f, indent=4)
        if self.args.save_per_iou:
            iou_josn_path = os.path.join(self.args.output_dir, 'iou.json')
            sorted_iou_rec = dict(sorted(iou_rec.items(), key=lambda x: self.safe_float(x[1]), reverse=True))
            with open(iou_josn_path, 'w') as f:
                score = json.dump(sorted_iou_rec, f, indent=4)
        return 
    
    def safe_float(self,v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0  # 发生错误时给个默认值，保证排序不崩溃

def main():
    parser = argparse.ArgumentParser(description="Training on OOS Dataset")

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--save_map', default=False,action="store_true")
    parser.add_argument('--save_per_iou', default=False, action="store_true")
    parser.add_argument('--use_slide', default=False, action="store_true")
    parser.add_argument('--num_workers', type=int)
    parser.add_argument("--cloud_path",type=str,default=None)
    parser.add_argument('--disable_fam', action='store_true')
    parser.add_argument('--disable_tta', action='store_true')
    parser.add_argument(
        '--sar_pretrained_path',
        type=str,
        default=None,
        help='Optional for evaluation because the trained checkpoint supplies the encoder weights.',
    )
    
    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.test()


if __name__ == "__main__":
    main()
