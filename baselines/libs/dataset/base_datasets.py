import numpy as np
import torch
from PIL import Image, ImageOps
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List

from dataset_api.zl_fabric import ZLImage

'''
FabricFinal RGB
mean:   0.428, 0.435,  0.457
std:    0.277,  0.274, 0.266
'''
MEAN_RGB = [0.428, 0.435, 0.457]
STD_RGB = [0.277, 0.274, 0.266]
MEAN_GRAY = [0.435, ]
STD_GRAY = [0.274, ]


class _LabelDataset(Dataset):
    def __init__(self, zl_imgs: List[ZLImage], normalization: str, color_mode: str, img_size: int, mean=None, std=None,
                 transform=None) -> None:
        self.zl_imgs = zl_imgs
        self.normalization = normalization
        self.img_size = img_size
        self.color_mode = color_mode
        if mean and std:
            self.mean = mean
            self.std = std
        else:
            if self.color_mode == 'gray':
                self.mean, self.std = MEAN_GRAY, STD_GRAY
            else:
                self.mean, self.std = MEAN_RGB, STD_RGB
        self.transform = transform

    def __len__(self):
        return len(self.zl_imgs)

    def __getitem__(self, index):
        zl_image = self.zl_imgs[index]
        name = zl_image.id
        label = zl_image.annotation()
        img = zl_image.image()
        if self.color_mode == 'gray':
            img = img.convert('L')
        if self.transform != None:
            img = self.transform(img)
        if self.normalization == 'zscore':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])(img)
        elif self.normalization == '[0,1]':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
            ])(img)
        elif self.normalization == '[-1,1]':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, ] * len(self.mean), std=[.5, ] * len(self.std)),
            ])(img)
        else:
            raise NotImplementedError
        return name, label, img


class _ObjectDataset(Dataset):
    def __init__(self, zl_imgs: List[ZLImage], normalization: str, color_mode: str, img_size: int, mean=None, std=None,
                 transform=None) -> None:
        self.zl_imgs = zl_imgs
        self.normalization = normalization
        self.img_size = img_size
        self.color_mode = color_mode
        if mean and std:
            self.mean = mean
            self.std = std
        else:
            if self.color_mode == 'gray':
                self.mean, self.std = MEAN_GRAY, STD_GRAY
            else:
                self.mean, self.std = MEAN_RGB, STD_RGB
        self.transform = transform

    def __len__(self):
        return len(self.zl_imgs)

    def __getitem__(self, index):
        zl_image = self.zl_imgs[index]
        name = zl_image.id
        label = zl_image.info()['defective']
        img = zl_image.image()
        bboxes = zl_image.annotation()
        if self.color_mode == 'gray':
            img = img.convert('L')
        if self.transform:
            img = self.transform(img)
        if self.normalization == 'zscore':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])(img)
        elif self.normalization == '[0,1]':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
            ])(img)
        elif self.normalization == '[-1,1]':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, ] * len(self.mean), std=[.5, ] * len(self.std)),
            ])(img)
        else:
            raise NotImplementedError
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        return name, label, img, bboxes


class _MaskDataset(Dataset):
    def __init__(self, zl_imgs: List[ZLImage], normalization: str, color_mode: str, img_size: int, mean=None, std=None,
                 transform=None) -> None:
        self.zl_imgs = zl_imgs
        self.normalization = normalization
        self.img_size = img_size
        self.color_mode = color_mode
        if mean and std:
            self.mean = mean
            self.std = std
        else:
            if self.color_mode == 'gray':
                self.mean, self.std = MEAN_GRAY, STD_GRAY
            else:
                self.mean, self.std = MEAN_RGB, STD_RGB
        self.transform = transform

    def __len__(self):
        return len(self.zl_imgs)

    def __getitem__(self, index):
        zl_image = self.zl_imgs[index]
        name = zl_image.id
        label = zl_image.info()['defective']
        img = zl_image.image()
        mask = zl_image.annotation(ann_type='mask')
        if self.color_mode == 'gray': img = img.convert('L')
        if img.mode == 'L' and self.color_mode == 'rgb': img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.normalization == 'zscore':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])(img)
        elif self.normalization == '[0,1]':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
            ])(img)
        elif self.normalization == '[-1,1]':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, ] * len(self.mean), std=[.5, ] * len(self.std)),
            ])(img)
        else:
            raise NotImplementedError
        mask = mask.convert('L').resize([self.img_size, self.img_size], resample=Image.NEAREST)
        mask = (np.array(mask) > 0)
        mask = torch.tensor(mask[None, ...], dtype=torch.float32)
        return name, label, img, mask


class _PairDataset(Dataset):
    def __init__(self, zl_imgs: List[ZLImage], normalization: str, color_mode: str, img_size: int,
                 mean=None, std=None, transform=None) -> None:
        self.zl_imgs = zl_imgs
        self.normalization = normalization
        self.img_size = img_size
        self.color_mode = color_mode
        if mean and std:
            self.mean = mean
            self.std = std
        else:
            if self.color_mode == 'gray':
                self.mean, self.std = MEAN_GRAY, STD_GRAY
            else:
                self.mean, self.std = MEAN_RGB, STD_RGB
        self.transform = transform

    def __len__(self):
        return len(self.zl_imgs)

    def __getitem__(self, index):
        zl_image = self.zl_imgs[index]
        name = zl_image.id
        label = zl_image.info()['defective']
        src_img = zl_image.image()
        assert label == 0
        if self.color_mode == 'gray':
            src_img = src_img.convert('L')
        tat_img = src_img.copy()
        if self.transform:
            src_img = self.transform(src_img)
        if self.normalization == 'zscore':
            tvf = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        elif self.normalization == '[0,1]':
            tvf = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
            ])
        elif self.normalization == '[-1,1]':
            tvf = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, ] * len(self.mean), std=[.5, ] * len(self.std)),
            ])
        else:
            raise NotImplementedError
        src_img = tvf(src_img)
        tat_img = tvf(tat_img)
        return name, label, src_img, tat_img


class _PatchDataset(Dataset):
    def __init__(self, zl_imgs: List[ZLImage], normalization: str, color_mode: str, img_size: int,
                 patch_size: int, patch_stride: int, mean=None, std=None, transform=None) -> None:
        self.zl_imgs = zl_imgs
        self.color_mode = color_mode
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.normalization = normalization
        if mean and std:
            self.mean = mean
            self.std = std
        else:
            if self.color_mode == 'gray':
                self.mean, self.std = MEAN_GRAY, STD_GRAY
            else:
                self.mean, self.std = MEAN_RGB, STD_RGB

    def __len__(self):
        return len(self.zl_imgs)

    def __getitem__(self, index):
        zl_image = self.zl_imgs[index]
        name = zl_image.id
        label = zl_image.info()['defective']
        img = zl_image.image()
        mask = zl_image.annotation()
        if self.color_mode == 'gray':
            img = img.convert('L')
        if self.normalization == '[0,1]' or self.normalization is None:
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
            ])(img)
        elif self.normalization == '[-1,1]':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, ] * len(self.mean), std=[.5, ] * len(self.std)),
            ])(img)
        elif self.normalization == 'zscore':
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])(img)
        elif self.normalization == 'histeq':
            assert self.color_mode == 'gray'
            img = ImageOps.equalize(img)
            img = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
            ])(img)
        else:
            raise NotImplementedError
        img = img.numpy()
        patch = self.extract_patches(img)
        if mask:
            mask = mask.convert('L').resize([self.img_size, self.img_size], resample=Image.NEAREST)
            mask = (np.array(mask) > 0).astype(dtype=np.float32)
            mask = mask[None, ...]
        return name, label, img, patch, mask

    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose([1, 2, 0])
        patch_size = tuple([self.patch_size] * 2)
        patch_stride = tuple([self.patch_stride] * 2)
        assert image.shape[0] >= patch_size[0] and image.shape[1] >= patch_size[1]
        channel = image.shape[2]
        assert channel in [1, 3]
        patch_size = patch_size + (channel,)
        patch_stride = patch_stride + (channel,)
        patch_indices_shape = ((np.array(image.shape) - np.array(patch_size)) //
                               np.array(patch_stride)) + 1
        shape = tuple(list(patch_indices_shape) + list(patch_size))
        patch_strides = image.strides
        slices = tuple([slice(None, None, st) for st in patch_stride])
        indexing_strides = image[slices].strides
        strides = tuple(list(indexing_strides) + list(patch_strides))
        patches = as_strided(image, shape=shape, strides=strides)
        patches = patches.squeeze(axis=2)
        patches = patches.transpose([0, 1, 4, 2, 3])
        return patches

    def merge_patches(self, patches: np.ndarray) -> np.ndarray:
        from .data_utils import merge_patches_to_image
        rec_image = merge_patches_to_image(patches=patches, patch_stride=self.patch_stride)
        return rec_image
