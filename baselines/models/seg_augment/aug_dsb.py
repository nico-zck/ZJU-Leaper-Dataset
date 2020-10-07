import random
import time
import warnings
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tvf

from baselines.libs.dataset import _DSBuilder, _MaskDataset
from dataset_api import ZLFabric


class AugDSB(_DSBuilder):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        cfg_dataset = cfg.DATASET
        self.normalization = cfg_dataset.normalization
        self.color_mode = cfg_dataset.color_mode
        self.img_size = cfg_dataset.image_size
        self.setting = cfg_dataset.setting
        self.zl_fabric = ZLFabric(dir=cfg_dataset.dataset_dir, fabric=cfg_dataset.fabric, setting=self.setting)
        assert self.normalization in ['[0,1]', '[-1,1]', 'zscore', 'histeq', ]
        if self.normalization == '[-1,1]':
            warnings.warn('Network using Sigmoid as output activation, you should use `[0,1]` normalization!')
        elif self.normalization == 'histeq':
            warnings.warn('histeq have problem to keep the background source and target image as same')

    def build_train(self) -> Tuple[Dataset, Dataset]:
        zlimgs_train_normal, zlimgs_train_defect, zlimgs_train_eval_normal, zlimgs_train_eval_defect \
            = self.zl_fabric.prepare_train()
        dt_train = AugDataset(good_zl_imgs=zlimgs_train_normal, bad_zl_imgs=zlimgs_train_defect,
                              normalization=self.normalization, color_mode=self.color_mode, img_size=self.img_size, )
        dt_train_eval = _MaskDataset(zl_imgs=zlimgs_train_eval_normal + zlimgs_train_eval_defect,
                                     normalization=self.normalization,
                                     color_mode=self.color_mode, img_size=self.img_size)
        return dt_train, dt_train_eval

    def build_train_dev(self) -> Tuple[Dataset, Dataset, Dataset]:
        zlimgs_folds = self.zl_fabric.prepare_k_fold(k_fold=5, shuffle=True)
        zlimgs_k_train_normal, zlimgs_k_train_defect, \
        zlimgs_k_train_eval_normal, zlimgs_k_train_eval_defect, \
        zlimgs_k_dev_normal, zlimgs_k_dev_defect = zlimgs_folds[0]
        dt_train = AugDataset(good_zl_imgs=zlimgs_k_train_normal, bad_zl_imgs=zlimgs_k_train_defect,
                              normalization=self.normalization, color_mode=self.color_mode, img_size=self.img_size)
        dt_train_eval = _MaskDataset(zl_imgs=zlimgs_k_train_eval_normal + zlimgs_k_train_eval_defect,
                                     normalization=self.normalization,
                                     color_mode=self.color_mode, img_size=self.img_size)
        dt_dev_eval = _MaskDataset(zl_imgs=zlimgs_k_dev_normal + zlimgs_k_dev_defect,
                                   normalization=self.normalization,
                                   color_mode=self.color_mode, img_size=self.img_size)
        return dt_train, dt_train_eval, dt_dev_eval

    def build_k_fold(self) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
        k_fold = self.cfg.DATASET.k_fold
        zlimgs_folds = self.zl_fabric.prepare_k_fold(k_fold=k_fold, shuffle=True)
        dt_train_list = []
        dt_train_eval_list = []
        dt_dev_list = []
        for zlimgs_k_train_normal, zlimgs_k_train_defect, \
            zlimgs_k_train_eval_normal, zlimgs_k_train_eval_defect, \
            zlimgs_k_dev_normal, zlimgs_k_dev_defect in zlimgs_folds:
            dt_train = AugDataset(good_zl_imgs=zlimgs_k_train_normal, bad_zl_imgs=zlimgs_k_train_defect,
                                  normalization=self.normalization, color_mode=self.color_mode, img_size=self.img_size)
            dt_train_eval = _MaskDataset(zl_imgs=zlimgs_k_train_eval_normal + zlimgs_k_train_eval_defect,
                                         normalization=self.normalization,
                                         color_mode=self.color_mode, img_size=self.img_size)
            dt_dev_eval = _MaskDataset(zl_imgs=zlimgs_k_dev_normal + zlimgs_k_dev_defect,
                                       normalization=self.normalization,
                                       color_mode=self.color_mode, img_size=self.img_size)
            dt_train_list.append(dt_train)
            dt_train_eval_list.append(dt_train_eval)
            dt_dev_list.append(dt_dev_eval)
        return dt_train_list, dt_train_eval_list, dt_dev_list

    def build_test(self) -> Dataset:
        zlimgs_test_normal, zlimgs_test_defect = self.zl_fabric.prepare_test()
        dt_test = _MaskDataset(zl_imgs=zlimgs_test_normal + zlimgs_test_defect,
                               normalization=self.normalization,
                               color_mode=self.color_mode, img_size=self.img_size)
        return dt_test


class AugDataset(Dataset):
    def __init__(self, good_zl_imgs, bad_zl_imgs, normalization, color_mode, img_size):
        self.good_zl_imgs = good_zl_imgs
        self.bad_zl_imgs = bad_zl_imgs
        self.normalization = normalization
        self.color_mode = color_mode
        self.image_size = img_size
        if self.color_mode == 'gray':
            self.mean = [0.435, ]
            self.std = [0.274, ]
        else:
            self.mean = [0.428, 0.435, 0.457]
            self.std = [0.277, 0.274, 0.266]
        self._init_defects()
        print('Dataset initialization is end.')

    def _init_defects(self):
        defect_imgs = []
        defect_masks = []
        for zl_image in self.bad_zl_imgs:
            img = zl_image.image()
            mask = zl_image.annotation()
            img = tvf.resize(img, self.image_size)
            mask = tvf.resize(mask, self.image_size, Image.NEAREST)
            defect_imgs.append(img)
            defect_masks.append(mask)
        self.defect_imgs = defect_imgs
        self.defect_masks = defect_masks
        self.color_transform = transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15)
        self.flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=[0, 180], translate=[0.25, 0.25], scale=[0.75, 1.25], shear=30),
        ])

    def _pair_image_transform(self, good_img):
        '''numpy.random will always produce same value when using multiprocessing, i.e. num_worker of Dataloader '''
        sample_index = random.randint(0, len(self.defect_imgs) - 1)
        bad_img = self.defect_imgs[sample_index]
        bad_img = self.color_transform(bad_img)
        bad_mask = self.defect_masks[sample_index]
        seed = time.time()
        random.seed(seed)
        bad_img = self.flip_transform(bad_img)
        random.seed(seed)
        bad_mask = self.flip_transform(bad_mask)
        bad_mask = np.array(bad_mask) / 255.
        if bad_img.mode != 'L':
            bad_mask = bad_mask[..., None]
        fused_img = np.array(bad_img) * bad_mask + np.array(good_img) * (1 - bad_mask)
        fused_img = Image.fromarray(np.uint8(fused_img))
        return fused_img, bad_mask

    def __len__(self):
        return len(self.good_zl_imgs)

    def __getitem__(self, index):
        zl_image = self.good_zl_imgs[index]
        name = zl_image.id
        label = zl_image.info()['defective']
        img = zl_image.image()
        img = tvf.resize(img, size=self.image_size)
        fused_img, bad_mask = self._pair_image_transform(good_img=img)
        tf = []
        if self.color_mode == 'gray':
            tf.append(transforms.Grayscale())
        if self.normalization == '[0,1]' or self.normalization is None:
            tf.append(transforms.ToTensor())
        elif self.normalization == '[-1,1]':
            tf.append(transforms.ToTensor())
            tf.append(transforms.Normalize(mean=[.5, ] * len(self.mean), std=[.5, ] * len(self.std)))
        elif self.normalization == 'zscore':
            tf.append(transforms.ToTensor())
            tf.append(transforms.Normalize(mean=self.mean, std=self.std))
        else:
            raise NotImplementedError
        tf = transforms.Compose(tf)
        bad_img = tf(fused_img)
        bad_mask = bad_mask.squeeze()[None, ...]
        bad_mask = torch.tensor(bad_mask, dtype=torch.float32)
        return name, label, bad_img, bad_mask,
