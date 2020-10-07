import json
import os
import socket
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from math import isnan

from pandas import DataFrame
from tensorboardX import SummaryWriter

from ..configurer import Configurer
from ..dataset_api import ZLEval


class _Evaluator(ABC):
    def __init__(self, cfg: Configurer):
        self.cfg = cfg

    @property
    def model_name(self):
        return self.cfg.TEMP.full_name

    def evaluating(self, model_dir, result_dict, subset, n_fold: int = None, **kwargs):
        print("evaluating results...")
        metrics_dict = OrderedDict()
        metrics_dict["model_name"] = self.model_name
        if "fabric" in self.cfg.DATASET: metrics_dict["pattern"] = self.cfg.DATASET.fabric
        if n_fold: metrics_dict["n-fold"] = n_fold
        temp_dict = OrderedDict()
        basic_metrics = self._basic_metrics(result_dict=result_dict, **kwargs)
        temp_dict.update(basic_metrics)
        other_metrics = self._other_metrics(result_dict=result_dict, **kwargs)
        temp_dict.update(other_metrics)
        temp_dict = {k: 0 if isnan(v) else v for k, v in temp_dict.items()}
        metrics_dict.update(temp_dict)
        with SummaryWriter(os.path.join(model_dir, 'log')) as writer:
            writer.add_hparams(hparam_dict={'set': subset},
                               metric_dict={'metric/' + k: v for k, v in temp_dict.items()},
                               name=f'metric_{subset}')
        if subset: metrics_dict["subset"] = subset
        metrics_dict["datetime"] = datetime.today().isoformat()
        metrics_dict["hostname"] = socket.gethostname()
        print(json.dumps(metrics_dict, indent=4))
        metrics_df = DataFrame(metrics_dict, index=[0])
        file_name = os.path.join(model_dir, self.model_name + '_metrics.csv')
        if not os.path.exists(file_name):
            metrics_df.to_csv(file_name, index=False)
        else:
            metrics_df.to_csv(file_name, mode='a', header=False, index=False)
        print('evaluation done.')
        return metrics_dict

    def epoch_evaluating(self, result_dict, **kwargs):
        metrics_dict = OrderedDict()
        basic_metrics = self._basic_metrics(result_dict=result_dict, **kwargs)
        metrics_dict.update(basic_metrics)
        other_metrics = self._other_metrics(result_dict=result_dict, **kwargs)
        metrics_dict.update(other_metrics)
        metrics_dict = {k: 0 if isnan(v) else v for k, v in metrics_dict.items()}
        return metrics_dict

    @abstractmethod
    def _basic_metrics(self, result_dict, **kwargs):
        pass

    def _other_metrics(self, result_dict, **kwargs):
        return {}


class ClsEvaluator(_Evaluator):
    pass


class DetEvaluator(_Evaluator):
    pass


class SegEvaluator(_Evaluator):
    pass


class ZLEvaluator(_Evaluator):
    def _basic_metrics(self, result_dict, **kwargs):
        binary_mask_target = result_dict["mask"]
        binary_mask_pred = result_dict["mask_pred"]
        zl_eval = ZLEval(binary_pixel_target=binary_mask_target, binary_pixel_pred=binary_mask_pred)
        basic_metrics = zl_eval.evaluate()
        return basic_metrics
