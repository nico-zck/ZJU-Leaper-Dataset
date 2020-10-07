from abc import ABC, abstractmethod
from typing import Tuple, List

from torch.utils.data import Dataset

from ..configurer import Configurer
from ..utils import join_str_list


class _DSBuilder(ABC):
    def __init__(self, cfg: Configurer):
        self.cfg = cfg
        cfg_dataset = cfg.DATASET
        if hasattr(cfg_dataset, "dataset_name"):
            self.dataset_name = cfg_dataset.dataset_name
        elif hasattr(cfg_dataset, "dataset_dir"):
            self.dataset_name = cfg_dataset.dataset_dir.split('/')[-1]
        else:
            self.dataset_name = None
        self.name = join_str_list(
            self.dataset_name,
            cfg_dataset.fabric,
        )

    @abstractmethod
    def build_train(self) -> Tuple[Dataset, Dataset]:
        dt_train, dt_train_eval = Dataset(), Dataset()
        return dt_train, dt_train_eval

    @abstractmethod
    def build_train_dev(self) -> Tuple[Dataset, Dataset, Dataset]:
        dt_train, dt_train_eval, dt_dev_eval = Dataset(), Dataset(), Dataset()
        return dt_train, dt_train_eval, dt_dev_eval

    @abstractmethod
    def build_k_fold(self) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
        dt_train_list, dt_train_eval_list, dt_dev_eval_list = [Dataset()], [Dataset()], [Dataset()]
        return dt_train_list, dt_train_eval_list, dt_dev_eval_list

    @abstractmethod
    def build_test(self) -> Dataset:
        dt_test_eval = Dataset()
        return dt_test_eval
