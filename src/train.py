import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter 
from torchinfo import summary # Summary the amount of parameters
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
# from torch.cuda.amp import autocast, GradScaler

from load_data.make_data_loader import MultimodalDamageAssessmentDatset
# from model.msi.MSI import MSI
# from model.unet.UNet import UNet
import model.mbdd.MBDD as MBDDUtils
from calc_utils.evaluate import ConfusionMatrix
from calc_utils.calc_loss import MBDDLoss


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, 'set_random_seed'):
        worker_info.dataset.set_random_seed(worker_seed)


class Trainer(object):

    # 初始化
    def __init__(self, args):

        self.args = args
        self.args.max_iters = self.args.max_iters // self.args.train_batch_size
        self.val_dataset = MultimodalDamageAssessmentDatset(self.args.val_dataset_path, self.args.val_data_name_list, type='val')
        val_generator = torch.Generator()
        val_generator.manual_seed(self.args.seed + 1)
        self.val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            prefetch_factor=16,
            worker_init_fn=seed_worker,
            generator=val_generator,
        )

        # ------------- Model -------------
        self.deep_model = MBDDUtils.MBDD(
            num_classes=1,
            use_fam=not self.args.disable_fam,
            sar_pretrained_path=self.args.sar_pretrained_path,
        )
        self.deep_model = self.deep_model.cuda()

        # -------------optimizer-------------
        self.optim = MBDDUtils.get_optim(self.deep_model,self.args.weight_decay,self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            threshold=0.001,
            threshold_mode='abs',
            mode='max', 
            patience=6,
            factor=0.8, 
            min_lr=1e-6
        )
        self.ema_model = AveragedModel(self.deep_model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        
        # ------------- Log Path -------------
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.model_param_path, args.dataset, args.model_type + '_' + now_str)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        run_config = {
            key: value for key, value in vars(self.args).items()
            if key not in {'train_data_name_list', 'val_data_name_list'}
        }
        with open(os.path.join(self.model_save_path, 'run_config.json'), 'w', encoding='utf-8') as file:
            json.dump(run_config, file, indent=2, ensure_ascii=False)
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_save_path, 'tensorboard'))

        # -------------calc utils:Loss and Scores-------------
        self.train_evaluator_clf = ConfusionMatrix(num_class=2,ignore_index=255)
        self.evaluator_loc = ConfusionMatrix(num_class=2,ignore_index=255)
        self.evaluator_clf = ConfusionMatrix(num_class=2,ignore_index=255)
        self.criterion = MBDDLoss(max_iters=self.args.max_iters,writer=self.writer).cuda()
        self.best_round={}
        self.best_round["damageIou"]=0.0



        # -------------resume-------------
        if args.resume is not None: 
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

    
    def _to_cuda(self, data):
        # 1-3 位是图像 (float)
        pre = data[0].cuda().float()
        post = data[1].cuda().float()
        sar = data[2].cuda().float()
        
        # 4-5 位是标签 (long)
        loc = data[3].cuda().long()
        clf = data[4].cuda().long()
        
        # 第 6 位是字符串 idx，保持原样（留在 CPU）
        idx = data[5] 
        
        return pre, post, sar, loc, clf, idx
    
    # def get_modal_drop_prob(self, batch_id):
    #     total_iters = self.args.max_iters  # 31250
    #     current_step = batch_id + 1
    #     progress = current_step / total_iters
    

    #     target_p_opt = 0.12
    #     target_p_sar = 0.05

    #     if progress <= 0.05 :
    #         p_opt = target_p_opt * (progress/0.05)
    #         p_sar = target_p_sar * (progress/0.05)
    #     elif 0.05 < progress < 0.85:
    #         p_opt = target_p_opt 
    #         p_sar = target_p_sar 
    #     else:
    #         p_opt = 0.0
    #         p_sar = 0.0

    #     return p_opt, p_sar

    def training(self):
        torch.cuda.empty_cache()
        damage_iou_val=0.0 # update lr
        self.train_evaluator_clf.reset()

        train_dataset = MultimodalDamageAssessmentDatset(self.args.train_dataset_path, self.args.train_data_name_list, type='train', max_iters=self.args.max_iters * self.args.train_batch_size,cloud_path=self.args.cloud_path)
        train_generator = torch.Generator()
        train_generator.manual_seed(self.args.seed)
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=16,
            worker_init_fn=seed_worker,
            generator=train_generator,
        )
        total = len(train_data_loader)
        # train
        self.deep_model.train()
        for batch_id, data in tqdm(enumerate(train_data_loader, 0), total=total,dynamic_ncols=True):
            # -------------warmup-------------
            if batch_id <= self.args.warmup_iters:
                warmup_ratio = batch_id / self.args.warmup_iters
                for param_group in self.optim.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * warmup_ratio
                current_lr = self.optim.param_groups[-1]['lr']
                self.writer.add_scalar("Loss/Learning_Rate", current_lr, batch_id+1)

            # -------------get data-------------
            pre_change_imgs, post_change_imgs, post_change_sar, labels_loc, labels_clf, _ = self._to_cuda(data)
            # skip the whole nodata label
            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue

            # -------------predict-------------
            self.optim.zero_grad()  
            # p_opt,p_sar=self.get_modal_drop_prob(batch_id=batch_id)
            output_loc, output_clf, align_loss, aux_damage, aux_sar = self.deep_model(pre_change_imgs,post_change_imgs,post_change_sar,labels_loc,throw_sar_p=0.2)
            final_loss,loc_loss,clf_loss = self.criterion(output_loc, output_clf, labels_loc, labels_clf, aux_damage, aux_sar, align_loss)
            
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), max_norm=1.0)
            self.optim.step()   
            if batch_id > self.args.warmup_iters:
                self.ema_model.update_parameters(self.deep_model) 

            # -------------loss and forward-------------
            with torch.no_grad():
                pred_clf = output_clf.detach().float() 
                output_clf_binary = (pred_clf > 0).byte().squeeze(1)
                self.train_evaluator_clf.add_batch(labels_clf.long(), output_clf_binary.long())

            if (batch_id + 1) % 10 == 0:
                tqdm.write(f'Iter is {batch_id + 1}, final loss is {final_loss.item()}, loc loss is {loc_loss.item()}, clf loss is {clf_loss.item()}, align loss is {align_loss.item()}')
                if (batch_id + 1) % 100 == 0:
                    damage_iou = self.train_evaluator_clf.Intersection_over_Union()[1]
                    self.writer.add_scalar('DamageIOU/train', damage_iou.item(), batch_id + 1)
                    self.train_evaluator_clf.reset()
                    for i, sca in enumerate(self.deep_model.fusion.sar_scales):
                        self.writer.add_scalar(f"Detail/Sar_scale_Layer_{i}", sca.abs().item(), batch_id + 1)
                    for i, gca in enumerate(self.deep_model.fusion.gate_scales):
                        self.writer.add_scalar(f"Detail/Diff_scale_Layer_{i}", gca.abs().item(), batch_id + 1)
                # -------------eval-------------
                if(batch_id +1)% 300 ==0:
                    self.deep_model.eval()
                    damage_iou_val =self.validation(batch_id,self.deep_model,'')
                    if (batch_id+1) > self.args.warmup_iters:
                        self.scheduler.step(damage_iou_val)
                        current_lr = self.optim.param_groups[-1]['lr']
                        self.writer.add_scalar("Loss/Learning_Rate", current_lr, batch_id+1)
                        if (batch_id+1) % 600==0 or (batch_id+1)>=self.args.max_iters*0.5:
                            self.validation(batch_id,self.ema_model,'EMA')
                    self.deep_model.train()
            
        print('The accuracy of the best round is ', self.best_round)
        self.writer.close()

    def validation(self,batch_id,model,state):
        print(f'---------starting {state} validation-----------')
        model.eval()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        torch.cuda.empty_cache()

        final_loss_val, loc_loss_val, clf_loss_val = 0, 0, 0

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.val_data_loader),total=len(self.val_data_loader),leave=False,dynamic_ncols=True):
                pre_change_imgs, post_change_imgs, post_change_sar, labels_loc, labels_clf, _ = self._to_cuda(data)
                
                output_loc, output_clf,_,_,_ = model(pre_change_imgs,post_change_imgs,post_change_sar)
                
                final_loss, loc_loss, clf_loss = self.criterion(output_loc, output_clf, labels_loc, labels_clf, state="val")

                final_loss_val += final_loss.item()
                loc_loss_val += loc_loss.item()
                clf_loss_val += clf_loss.item()
                
                output_loc_binary = (output_loc.detach().float() > 0).byte().squeeze(1)
                output_clf_binary = (output_clf.detach().float()> 0).byte().squeeze(1)

                self.evaluator_loc.add_batch(labels_loc.long(), output_loc_binary.long())
                self.evaluator_clf.add_batch(labels_clf.long(), output_clf_binary.long())
        
        avg_loss_val = final_loss_val / len(self.val_data_loader)
        avg_loc_loss = loc_loss_val / len(self.val_data_loader)
        avg_clf_loss = clf_loss_val / len(self.val_data_loader)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        loc_OA = self.evaluator_loc.Pixel_Accuracy()
        loc_mIoU = self.evaluator_loc.Mean_Intersection_over_Union()
        build_iou = self.evaluator_loc.Intersection_over_Union()[1]
        
        clf_f1_score = self.evaluator_clf.Pixel_F1_score()
        clf_OA = self.evaluator_clf.Pixel_Accuracy()
        damage_iou = self.evaluator_clf.Intersection_over_Union()[1]
        clf_mIoU = self.evaluator_clf.Mean_Intersection_over_Union()
        clf_P= self.evaluator_clf.Pixel_Precision_Rate()
        clf_R= self.evaluator_clf.Pixel_Recall_Rate()
        
        self.writer.add_scalar(f"Loss/{state}Val",avg_loss_val,batch_id+1)
        self.writer.add_scalar(f"Loss/{state}val_clf",avg_clf_loss,batch_id+1)
        self.writer.add_scalar(f'DamageIOU/{state}val', damage_iou.item(), batch_id + 1)
        self.writer.add_scalar(f"DamageIOU/{state}val_clf_P",clf_P.item(),batch_id+1)
        self.writer.add_scalar(f"DamageIOU/{state}val_clf_R",clf_R.item(),batch_id+1)

        if state == '':
            self.writer.add_scalar("LossDetail/val_loc",avg_loc_loss,batch_id+1)
            self.writer.add_scalar('Scores/Loc_F1', loc_f1_score.item(), batch_id + 1)
            self.writer.add_scalar('Scores/Loc_mIoU', loc_mIoU.item(), batch_id + 1)
            self.writer.add_scalar('Scores/Building_Iou', build_iou.item(), batch_id + 1)
            self.writer.add_scalar('Scores/CLF_F1', clf_f1_score.item(), batch_id + 1)
            self.writer.add_scalar('Scores/CLF_mIoU', clf_mIoU.item(), batch_id + 1)
        # uopdate the best model
        if damage_iou.item()> self.best_round['damageIou']:
            torch.save(model.state_dict(), os.path.join(self.model_save_path, f'best_model.pth'))
            self.best_round = {
                'best iter': batch_id + 1,
                'EMA': state=="EMA",
                'loc f1': loc_f1_score.item(),
                'loc OA': loc_OA.item(),
                'loc mIoU': loc_mIoU.item(),
                'buildIou': build_iou.item(),
                'clf f1': clf_f1_score.item(),
                'clf OA': clf_OA.item(),
                'clf mIoU': clf_mIoU.item(),
                'damageIou': damage_iou.item()
            }
            
        tqdm.write(f'LOC:F1-score is {loc_f1_score.item()}, OA is {loc_OA.item()}, mIoU is {loc_mIoU.item()}, IoU of building is {build_iou.item()}')
        tqdm.write(f'CLF:F1-score is {clf_f1_score.item()}, OA is {clf_OA.item()}, mIoU is {clf_mIoU.item()}, IoU of damage is {damage_iou.item()},P is {clf_P.item()},R is {clf_R.item()}')
        
        return damage_iou.item()

def main():
    parser = argparse.ArgumentParser(description="Training on OOS Dataset")


    parser.add_argument('--dataset', type=str, default='OOS')
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--train_data_list_path', type=str)
    parser.add_argument('--val_dataset_path', type=str)
    parser.add_argument('--val_data_list_path', type=str)
    parser.add_argument('--cloud_path', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=500000)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_param_path', type=str, default='./saved_weights')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.03) # 5e-3
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_fam', action='store_true')
    parser.add_argument(
        '--sar_pretrained_path',
        type=str,
        default=None,
        help='Converted BigEarthNet Sentinel-1 ResNet-18 checkpoint. See docs/PRETRAINED_WEIGHTS.md.',
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    with open(args.train_data_list_path, "r") as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = train_data_name_list

    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list

    trainer = Trainer(args)
    trainer.training()

if __name__ == "__main__":
    main()
