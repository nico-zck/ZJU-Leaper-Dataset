import random

import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Sampler

from .base_datasets import MEAN_GRAY, MEAN_RGB, STD_GRAY, STD_RGB


class SubsetRandomSampler(Sampler):
    def __init__(self, dataset, subset_ratio=1.0):
        super().__init__(data_source=dataset)
        assert 0 < subset_ratio <= 1
        num_sample = len(dataset)
        random.seed(123)
        if subset_ratio != 1:
            self.indices = random.sample(range(num_sample), int(num_sample * subset_ratio))
        else:
            self.indices = np.arange(num_sample)

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def restore_image_normalization(imgs: np.ndarray, normalization: str):
    nc = imgs.shape[1]
    if normalization == 'zscore':
        if nc == 3:
            mean = np.array(MEAN_RGB, dtype=imgs.dtype).reshape(1, 3, 1, 1)
            std = np.array(STD_RGB, dtype=imgs.dtype).reshape(1, 3, 1, 1)
            imgs = (imgs * std) + mean
        elif nc == 1:
            mean = np.array(MEAN_GRAY, dtype=imgs.dtype)
            std = np.array(STD_GRAY, dtype=imgs.dtype)
            imgs = (imgs * std) + mean
        else:
            raise NotImplementedError
    elif normalization == '[-1,1]':
        imgs = imgs * 0.5 + 0.5
        imgs = np.clip(imgs, a_min=0, a_max=1)
    elif normalization == '[0,1]' or normalization == 'histeq':
        imgs = np.clip(imgs, a_min=0, a_max=1)
    else:
        print(imgs.min(), imgs.max())
        raise NotImplementedError
    return imgs


def extract_patches_from_image(image: np.ndarray, patch_size: int, patch_stride: int):
    patch_size = tuple([patch_size] * 2)
    patch_stride = tuple([patch_stride] * 2)
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
    return patches


def merge_patches_to_image(patches: np.ndarray, patch_stride: int) -> np.ndarray:
    patches = patches.transpose([0, 1, 3, 4, 2])
    patch_stride = tuple([patch_stride] * 2)
    arrange_shape = patches.shape[0:2]
    patch_size = patches.shape[2:4]
    channel = patches.shape[-1]
    patches = patches.reshape([-1, *patches.shape[2:5]])

    def _get_merged_shape(patches_shape, patch_size, patch_stride):
        patch_rows = patches_shape[0]
        patch_cols = patches_shape[1]
        out_shape = (
            patch_rows * patch_stride[0] + (patch_size[0] - patch_stride[0]),
            patch_cols * patch_stride[1] + (patch_size[1] - patch_stride[1])
        )
        return out_shape

    image_size = _get_merged_shape(arrange_shape, patch_size, patch_stride)
    rec_image = np.zeros(np.append(image_size, channel))
    rec_image_divide = np.zeros(image_size)
    img_h, img_w = image_size
    index = 0
    for hs, he in zip(range(0, img_h, patch_stride[0]), range(patch_size[0], img_h + 1, patch_stride[0])):
        for ws, we in zip(range(0, img_w, patch_stride[1]), range(patch_size[1], img_w + 1, patch_stride[1])):
            rec_image[hs:he, ws:we, :] += patches[index, :]
            rec_image_divide[hs:he, ws:we] += 1
            index += 1
    rec_image /= rec_image_divide[..., None]
    rec_image = rec_image.transpose([2, 0, 1])
    return rec_image
