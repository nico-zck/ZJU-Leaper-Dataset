import warnings
from typing import Tuple, List

from torch.utils.data import Dataset

from baselines.libs.configurer import Configurer
from baselines.libs.dataset import _DSBuilder, _PairDataset, _MaskDataset
from dataset_api import ZLFabric


class PairAndMaskDSB(_DSBuilder):
    def __init__(self, cfg: Configurer):
        super().__init__(cfg)
        cfg_dataset = cfg.DATASET
        self.normalization = cfg_dataset.normalization
        self.color_mode = cfg_dataset.color_mode
        self.img_size = cfg_dataset.image_size
        assert 'setting' not in cfg_dataset
        self.zl_fabric_src_domain = ZLFabric(dir=cfg_dataset.dataset_dir, fabric=cfg_dataset.fabric, setting='setting1')
        self.zl_fabric_tgt_domain = ZLFabric(dir=cfg_dataset.dataset_dir, fabric=cfg_dataset.fabric, setting='setting2')
        assert self.normalization in ['[0,1]', '[-1,1]', 'zscore', 'histeq', ]
        if self.normalization == '[-1,1]':
            warnings.warn('Network using Sigmoid as output activation, you should use `[0,1]` normalization!')
        elif self.normalization == 'histeq':
            warnings.warn('histeq have problem to keep the background source and target image as same')

    def build_train(self) -> Tuple[List[Dataset], Dataset]:
        zlimgs_train_normal, _, _, _ = self.zl_fabric_src_domain.prepare_train()
        dt_train_src = _PairDataset(zl_imgs=zlimgs_train_normal, normalization=self.normalization,
                                    color_mode=self.color_mode, img_size=self.img_size)
        _, zlimgs_train_defect, zlimgs_train_eval_normal, zlimgs_train_eval_defect \
            = self.zl_fabric_tgt_domain.prepare_train()
        dt_train_tgt = _MaskDataset(zl_imgs=zlimgs_train_defect, normalization=self.normalization,
                                    color_mode=self.color_mode, img_size=self.img_size)
        dt_train_eval = _MaskDataset(zl_imgs=zlimgs_train_eval_normal + zlimgs_train_eval_defect,
                                     normalization=self.normalization,
                                     color_mode=self.color_mode, img_size=self.img_size)
        return [dt_train_src, dt_train_tgt], dt_train_eval

    def build_train_dev(self) -> Tuple[List[Dataset], Dataset, Dataset]:
        zlimgs_folds_src = self.zl_fabric_src_domain.prepare_k_fold(k_fold=5, shuffle=True)
        zlimgs_k_train_normal, _, _, _, _, _ = zlimgs_folds_src[0]
        dt_train_src = _PairDataset(zl_imgs=zlimgs_k_train_normal, normalization=self.normalization,
                                    color_mode=self.color_mode, img_size=self.img_size)
        zlimgs_folds_tgt = self.zl_fabric_tgt_domain.prepare_k_fold(k_fold=5, shuffle=True)
        _, zlimgs_k_train_defect, \
        zlimgs_k_train_eval_normal, zlimgs_k_train_eval_defect, \
        zlimgs_k_dev_normal, zlimgs_k_dev_defect = zlimgs_folds_tgt[0]
        dt_train_tgt = _MaskDataset(zl_imgs=zlimgs_k_train_defect, normalization=self.normalization,
                                    color_mode=self.color_mode, img_size=self.img_size)
        dt_train_eval = _MaskDataset(zl_imgs=zlimgs_k_train_eval_normal + zlimgs_k_train_eval_defect,
                                     normalization=self.normalization,
                                     color_mode=self.color_mode, img_size=self.img_size)
        dt_dev_eval = _MaskDataset(zl_imgs=zlimgs_k_dev_normal + zlimgs_k_dev_defect,
                                   normalization=self.normalization,
                                   color_mode=self.color_mode, img_size=self.img_size)
        return [dt_train_src, dt_train_tgt], dt_train_eval, dt_dev_eval

    def build_k_fold(self):
        k_fold = self.cfg.DATASET.k_fold
        assert k_fold > 1
        zlimgs_folds_src = self.zl_fabric_src_domain.prepare_k_fold(k_fold=k_fold, shuffle=True)
        zlimgs_folds_tgt = self.zl_fabric_tgt_domain.prepare_k_fold(k_fold=k_fold, shuffle=True)
        dt_train_list = []
        dt_train_eval_list = []
        dt_dev_list = []
        for zlimgs_k_fold_src, zlimgs_k_fold_tgt in zip(zlimgs_folds_src, zlimgs_folds_tgt):
            zlimgs_k_train_normal, _, _, _, _, _ = zlimgs_k_fold_src
            dt_train_src = _PairDataset(zl_imgs=zlimgs_k_train_normal, normalization=self.normalization,
                                        color_mode=self.color_mode, img_size=self.img_size)
            _, zlimgs_k_train_defect, \
            zlimgs_k_train_eval_normal, zlimgs_k_train_eval_defect, \
            zlimgs_k_dev_normal, zlimgs_k_dev_defect = zlimgs_k_fold_tgt
            dt_train_tgt = _MaskDataset(zl_imgs=zlimgs_k_train_defect, normalization=self.normalization,
                                        color_mode=self.color_mode, img_size=self.img_size)
            dt_train_eval = _MaskDataset(zl_imgs=zlimgs_k_train_eval_normal + zlimgs_k_train_eval_defect,
                                         normalization=self.normalization,
                                         color_mode=self.color_mode, img_size=self.img_size)
            dt_dev_eval = _MaskDataset(zl_imgs=zlimgs_k_dev_normal + zlimgs_k_dev_defect,
                                       normalization=self.normalization,
                                       color_mode=self.color_mode, img_size=self.img_size)
            dt_train_list.append([dt_train_src, dt_train_tgt])
            dt_train_eval_list.append(dt_train_eval)
            dt_dev_list.append(dt_dev_eval)
        return dt_train_list, dt_train_eval_list, dt_dev_list

    def build_test(self) -> Dataset:
        zlimgs_test_normal, zlimgs_test_defect = self.zl_fabric_tgt_domain.prepare_test()
        dt_test = _MaskDataset(zl_imgs=zlimgs_test_normal + zlimgs_test_defect,
                               normalization=self.normalization,
                               color_mode=self.color_mode, img_size=self.img_size)
        return dt_test
