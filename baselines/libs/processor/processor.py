from abc import ABC, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from .processing_utils import block_mean
from .threshold_helper import ThresholdHelper
from ..configurer import Configurer
from ..dataset.data_utils import restore_image_normalization


class _Processor(ABC):
    def __init__(self, cfg: Configurer):
        self.cfg = cfg

    def epoch_processing(self, result_dict: dict):
        result_dict = self._post_processing(result_dict=result_dict)
        return result_dict

    def processing(self, result_dict: dict, **kwargs):
        print("processing results...")
        result_dict = self._post_processing(result_dict=result_dict)
        print("processing done.")
        return result_dict

    @abstractmethod
    def _post_processing(self, result_dict: dict, **kwargs):
        pass


class ClsProcessor(_Processor):
    pass


class DetProcessor(_Processor):
    pass


class SegProcessor(_Processor):
    def _post_processing(self, result_dict: dict, **kwargs):
        raise NotImplementedError


class ImageRecProcessor(_Processor):
    def _post_processing(self, result_dict: dict, **kwargs):
        img_input = result_dict["image"]
        img_mask = result_dict["mask"]
        img_pred = result_dict["image_pred"]
        img_recon_error = (img_input - img_pred) ** 2
        img_post = np.mean(img_recon_error, axis=1)
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = \
                [executor.submit(block_mean, pred, block_size=8) for pred in img_post]
            img_post = []
            for f in tqdm(futures, desc='post-processor', dynamic_ncols=True):
                img_post.append(f.result())
        img_post = np.array(img_post)
        img_post = (img_post - img_post.min()) / (img_post.max() - img_post.min())
        img_post = img_post.squeeze()
        binary_target = img_mask.squeeze().astype(np.bool)
        helper = ThresholdHelper(img_post, binary_target, metric='dice')
        best_thr, max_value = helper.get_best_threshold()
        binary_pred = (img_post > best_thr)
        print('best threshold: %g, max value: %g' % (best_thr, max_value))
        result_dict["score_pred"] = img_post.squeeze()
        result_dict["mask_pred"] = binary_pred.squeeze()
        result_dict["mask"] = binary_target.squeeze()
        result_dict["image"] = \
            restore_image_normalization(imgs=result_dict["image"], normalization=self.cfg.DATASET.normalization)
        result_dict["image_pred"] = \
            restore_image_normalization(imgs=result_dict["image_pred"], normalization=self.cfg.DATASET.normalization)
        return result_dict


class ScoreMapProcessor(_Processor):
    def _post_processing(self, result_dict: dict, **kwargs):
        mask = result_dict["mask"]
        raw_pred = result_dict["raw_pred"]
        if mask.dtype != np.bool:
            assert np.unique(mask).size == 2
            mask = mask.astype(np.bool)
        score_pred = (raw_pred - raw_pred.min()) / (raw_pred.max() - raw_pred.min())
        helper = ThresholdHelper(score_pred, mask, metric='dice')
        best_thr, max_value = helper.get_best_threshold()
        mask_pred = (score_pred > best_thr)
        print('best threshold: %g, max value: %g' % (best_thr, max_value))
        result_dict["mask"] = mask.squeeze()
        result_dict["score_pred"] = score_pred.squeeze()
        result_dict["mask_pred"] = mask_pred.squeeze()
        result_dict["image"] = \
            restore_image_normalization(imgs=result_dict["image"], normalization=self.cfg.DATASET.normalization)
        return result_dict


class BinaryMapProcessor(_Processor):
    def _post_processing(self, result_dict: dict, **kwargs):
        mask = result_dict["mask"]
        mask_pred = result_dict["mask_pred"]
        result_dict["image"] = \
            restore_image_normalization(imgs=result_dict["image"], normalization=self.cfg.DATASET.normalization)
        return result_dict
