import os
import numpy as np
from torch.utils.data import Dataset
import load_data.aug_utils as imutils
from load_data.aug_utils  import AugUtils,CloudAugmentor
import random
import tifffile as tiff

def data_loader(path):
    img=tiff.imread(path)
    return np.array(img, np.uint8)


class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, type='train', data_loader=data_loader, suffix='.tif',max_iters=None,cloud_path=None):
        self.dataset_path = dataset_path

        self.data_list = data_list
        random.shuffle(self.data_list)

        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix
        self.augtiles=AugUtils(cloud_path=cloud_path)
        
        self.iters = len(data_list)
        if max_iters is not None:
            self.iters = max_iters
        
    
    def __len__(self):
        return self.iters

    def set_random_seed(self, seed):
        self.augtiles.set_random_seed(seed)

    def __transforms(self, aug, pre_img, post_img, post_sar, label):
        
        if aug:
            pre_img, post_img, post_sar, label= self.augtiles(pre_img, post_img, post_sar, label)

        pre_img = imutils.normalize_img(pre_img) 
        post_img = imutils.normalize_img(post_img)
        post_sar = imutils.normalize_sar(post_sar)

        pre_img = np.transpose(pre_img, (2, 0, 1))
        post_img = np.transpose(post_img, (2, 0, 1))
        post_sar = np.transpose(post_sar, (2, 0, 1))

        return pre_img, post_img,post_sar, label

    def __getitem__(self, fact_index):
        index = fact_index % len(self.data_list)

        pre_path = os.path.join(self.dataset_path, 'pre-event-opt', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event-opt', self.data_list[index] + '_post_disaster'  + self.suffix)
        sar_path=os.path.join(self.dataset_path, 'post-event-sar', self.data_list[index] + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)[:,:,0:3] 
        post_sar = self.loader(sar_path)
        post_sar = np.expand_dims(post_sar, axis=2)[:, :, 0:1]
        clf_label = self.loader(label_path)


        if 'train' in self.data_pro_type:
            pre_img, post_img, post_sar,clf_label = self.__transforms(True, pre_img, post_img, post_sar, clf_label)
        else:
            pre_img, post_img, post_sar,clf_label = self.__transforms(False, pre_img, post_img, post_sar, clf_label)
            clf_label = np.asarray(clf_label)

        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        clf_label[clf_label == 1] = 0
        clf_label[clf_label == 2] = 255
        clf_label[clf_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, post_sar, loc_label, clf_label, data_idx

class MultimodalDamageAssessmentDatset_Inference(Dataset):
    def __init__(self, dataset_path, data_list, type='train', data_loader=data_loader, suffix='.tif',max_iters=None,cloud_path=None):
        self.dataset_path = dataset_path

        self.data_list = data_list

        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix
        self.cloud_aug=None
        if cloud_path is not None:
            self.cloud_aug=CloudAugmentor(cloud_dir=cloud_path)
        
        self.base_seed=25
        
        self.iters = len(data_list)
        if max_iters is not None:
            self.iters = max_iters
        
    
    def __len__(self):
        return self.iters

    def __transforms(self, pre_img, post_img, post_sar, label, state):
        pre_img = imutils.normalize_img(pre_img) 
        post_img = imutils.normalize_img(post_img)
        post_sar = imutils.normalize_sar(post_sar)

        pre_img = np.transpose(pre_img, (2, 0, 1))
        post_img = np.transpose(post_img, (2, 0, 1))
        post_sar = np.transpose(post_sar, (2, 0, 1))

        return pre_img, post_img,post_sar, label

    def __getitem__(self, fact_index):
        index = fact_index % len(self.data_list)

        pre_path = os.path.join(self.dataset_path, 'pre-event-opt', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event-opt', self.data_list[index] + '_post_disaster'  + self.suffix)
        sar_path=os.path.join(self.dataset_path, 'post-event-sar', self.data_list[index] + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)[:,:,0:3] 
        post_sar = self.loader(sar_path)
        post_sar = np.expand_dims(post_sar, axis=2)[:, :, 0:1]
        clf_label = self.loader(label_path)



        seed=self.base_seed+index
        local_rng = random.Random(seed)
        if local_rng.random()>0.7 and self.cloud_aug is not None:
            post_img=self.cloud_aug.apply_cloud(post_img,global_alpha=local_rng.uniform(0.7,1.0),rand=local_rng)
        pre_img, post_img, post_sar,clf_label = self.__transforms(pre_img, post_img, post_sar, clf_label,'val')

        clf_label = np.asarray(clf_label)

        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        clf_label[clf_label == 1] = 0
        clf_label[clf_label == 2] = 255
        clf_label[clf_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, post_sar, loc_label, clf_label, data_idx


