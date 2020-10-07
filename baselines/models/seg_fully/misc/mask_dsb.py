from typing import Tuple, List
from typing import Tuple, List

from torch.utils.data import Dataset
from torch.utils.data import Dataset

from baselines.libs.configurer import Configurer
from baselines.libs.configurer import Configurer
from baselines.libs.dataset import _MaskDataset, _DSBuilder
from baselines.libs.dataset import _MaskDataset, _DSBuilder
from dataset_api import ZLFabric
from dataset_api import ZLFabric


class MaskDSB(_DSBuilder):
    def __init__(self, cfg: Configurer, TrainDataset=_MaskDataset, EvalDataset=_MaskDataset, rnd_seed=None):
        super().__init__(cfg)
        cfg_dataset = cfg.DATASET
        self.normalization = cfg_dataset.normalization
        self.color_mode = cfg_dataset.color_mode
        self.img_size = cfg_dataset.image_size
        self.setting = cfg_dataset.setting
        self.zl_fabric = ZLFabric(dir=cfg_dataset.dataset_dir, fabric=cfg_dataset.fabric, setting=self.setting,
                                  seed=rnd_seed)
        self.TrainDS = TrainDataset
        self.EvalDS = EvalDataset

    def build_train(self) -> Tuple[Dataset, Dataset]:
        zlimgs_train_normal, zlimgs_train_defect, zlimgs_train_eval_normal, zlimgs_train_eval_defect \
            = self.zl_fabric.prepare_train()
        dt_train = self.TrainDS(zl_imgs=zlimgs_train_defect, normalization=self.normalization,
                                color_mode=self.color_mode, img_size=self.img_size)
        dt_train_eval = self.EvalDS(zl_imgs=zlimgs_train_eval_normal + zlimgs_train_eval_defect,
                                    normalization=self.normalization, color_mode=self.color_mode,
                                    img_size=self.img_size)
        return dt_train, dt_train_eval

    def build_train_dev(self) -> Tuple[Dataset, Dataset, Dataset]:
        zlimgs_folds = self.zl_fabric.prepare_k_fold(k_fold=5, shuffle=True)
        zlimgs_k_train_normal, zlimgs_k_train_defect, \
        zlimgs_k_train_eval_normal, zlimgs_k_train_eval_defect, \
        zlimgs_k_dev_normal, zlimgs_k_dev_defect = zlimgs_folds[0]
        dt_train = self.TrainDS(zl_imgs=zlimgs_k_train_defect, normalization=self.normalization,
                                color_mode=self.color_mode, img_size=self.img_size)
        dt_train_eval = self.EvalDS(zl_imgs=zlimgs_k_train_eval_normal + zlimgs_k_train_eval_defect,
                                    normalization=self.normalization,
                                    color_mode=self.color_mode, img_size=self.img_size)
        dt_dev_eval = self.EvalDS(zl_imgs=zlimgs_k_dev_normal + zlimgs_k_dev_defect,
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
            dt_train = self.TrainDS(zl_imgs=zlimgs_k_train_defect, normalization=self.normalization,
                                    color_mode=self.color_mode, img_size=self.img_size)
            dt_train_eval = self.EvalDS(zl_imgs=zlimgs_k_train_eval_normal + zlimgs_k_train_eval_defect,
                                        normalization=self.normalization,
                                        color_mode=self.color_mode, img_size=self.img_size)
            dt_dev_eval = self.EvalDS(zl_imgs=zlimgs_k_dev_normal + zlimgs_k_dev_defect,
                                      normalization=self.normalization,
                                      color_mode=self.color_mode, img_size=self.img_size)
            dt_train_list.append(dt_train)
            dt_train_eval_list.append(dt_train_eval)
            dt_dev_list.append(dt_dev_eval)
        return dt_train_list, dt_train_eval_list, dt_dev_list

    def build_test(self) -> Dataset:
        zlimgs_test_normal, zlimgs_test_defect = self.zl_fabric.prepare_test()
        dt_test = self.EvalDS(zl_imgs=zlimgs_test_normal + zlimgs_test_defect,
                              normalization=self.normalization,
                              color_mode=self.color_mode, img_size=self.img_size)
        return dt_test
