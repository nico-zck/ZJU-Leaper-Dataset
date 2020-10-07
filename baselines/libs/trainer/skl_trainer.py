import pickle

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .trainer import _Trainer
from ..configurer import Configurer
from ..core_model import _SKLCoreModel
from ..dataset import _PatchDataset


class SKLTrainer(_Trainer):
    def __init__(self, cfg: Configurer, core_model: _SKLCoreModel):
        super().__init__(cfg, core_model)
        self.core_model = core_model

    def save_model(self, model_dir: str, is_best: bool = False, epoch: int = None):
        model_path = model_dir + '/model.pkl'
        state_dict = self.core_model.dump_core_model()
        with open(model_path, 'wb') as f:
            pickle.dump(state_dict, f)

    def load_model(self, model_dir: str, is_best: bool = False, epoch: int = None):
        model_path = model_dir + '/model.pkl'
        with open(model_path, 'rb') as f:
            state_dict = pickle.load(f)
        self.core_model.load_core_model(state_dict=state_dict)

    def reset_model(self):
        self.core_model.init_core_model()

    def model_train(self, train_dataset: Dataset, model_dir: str, **kwargs):
        dl = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
        all_data = []
        for batch_data in dl:
            all_data.append(batch_data)
        self.core_model.fit(train_data=all_data)

    def model_train_dev(self, train_dataset: Dataset, dev_dataset: Dataset, model_dir: str,
                        epoch_dev_eval: bool = False, **kwargs):
        assert epoch_dev_eval is False
        dl = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
        all_data = []
        for batch_data in dl:
            all_data.append(batch_data)
        self.core_model.fit(train_data=all_data)

    def model_eval(self, eval_dataset: _PatchDataset, eval_set: str, model_dir: str, **kwargs):
        dl = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)
        results = []
        for batch_data in tqdm(dl, desc='testing', dynamic_ncols=True):
            batch_result = self.core_model.batch_predict(batch_data=batch_data)
            results.append(batch_result)
        results = pd.DataFrame(results).to_dict(orient='list')
        results = {k: np.concatenate(v) for k, v in results.items()}
        return results
