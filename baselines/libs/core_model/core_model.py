from abc import ABC, abstractmethod

from ..configurer import Configurer


class _CoreModel(ABC):
    def __init__(self, cfg: Configurer):
        self.cfg = cfg

    @abstractmethod
    def dump_core_model(self) -> dict:
        pass

    @abstractmethod
    def load_core_model(self, state_dict: dict):
        pass

    @abstractmethod
    def init_core_model(self):
        self.model = None
