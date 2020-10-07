from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from ..configurer import Configurer
from ..core_model.core_model import _CoreModel


class _Trainer(ABC):
    def __init__(self, cfg: Configurer, core_model: _CoreModel):
        self.cfg = cfg
        self.core_model = core_model

    @property
    def name(self):
        return self.core_model.__class__.__name__

    @abstractmethod
    def reset_model(self):
        pass

    @abstractmethod
    def save_model(self, model_dir: str, is_best: bool = False, epoch: int = None):
        assert [is_best, epoch is not None].count(True) <= 1
        print('saving model: ', end='')
        if is_best:
            print('the best model...')
        if epoch:
            print(f'the model at epoch-{epoch}...')
        else:
            print('the last model...')

    @abstractmethod
    def load_model(self, model_dir: str, is_best: bool = False, epoch: int = None):
        assert [is_best, epoch is not None].count(True) <= 1
        print('loading model: ', end='')
        if is_best:
            print('the best model...')
        if epoch:
            print(f'the model at epoch-{epoch}...')
        else:
            print('the last model...')

    @abstractmethod
    def model_train(self, train_dataset: Dataset, model_dir: str, **kwargs):
        pass

    @abstractmethod
    def model_train_dev(self, train_dataset: Dataset, dev_dataset: Dataset,
                        model_dir: str, epoch_dev_eval: bool = False, **kwargs):
        pass

    @abstractmethod
    def model_eval(self, eval_dataset: Dataset, eval_set: str, model_dir: str, **kwargs):
        pass
