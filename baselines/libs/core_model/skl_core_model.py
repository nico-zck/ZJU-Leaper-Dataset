from abc import abstractmethod

from .core_model import _CoreModel
from ..configurer import Configurer


class _SKLCoreModel(_CoreModel):
    def __init__(self, cfg: Configurer):
        super().__init__(cfg)

    @abstractmethod
    def dump_core_model(self) -> dict:
        pass

    @abstractmethod
    def load_core_model(self, state_dict: dict):
        pass

    @abstractmethod
    def init_core_model(self):
        pass

    @abstractmethod
    def fit(self, train_data: list, **kwargs):
        pass

    @abstractmethod
    def batch_predict(self, batch_data, **kwargs):
        pass
