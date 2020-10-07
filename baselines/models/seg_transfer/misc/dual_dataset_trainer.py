import os
import shutil
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from baselines.libs.configurer import Configurer
from baselines.libs.core_model import _TorchCoreModel
from baselines.libs.trainer import TorchTrainer


class DualDatasetTrainer(TorchTrainer):
    def __init__(self, cfg: Configurer, core_model: _TorchCoreModel):
        super().__init__(cfg, core_model)
        self.epoch_s1, self.epoch_s2 = self.epochs

    def model_train(self, train_dataset: List[Dataset], model_dir: str, **kwargs):
        log_dir = os.path.join(model_dir, 'log')
        if self.cur_epoch == 0: shutil.rmtree(log_dir, ignore_errors=True)
        writer = SummaryWriter(log_dir, flush_secs=60)
        num_workers = kwargs.get('num_workers', 8)
        dt_train_src, dt_train_tgt = train_dataset
        dl_train_src = DataLoader(dt_train_src, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        dl_train_tgt = DataLoader(dt_train_tgt, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.core_model.train()
        for epoch in range(self.cur_epoch + 1, self.epoch_s1 + self.epoch_s2 + 1):
            epoch_start_time = time.time()
            self.cur_epoch = epoch
            running_losses = []
            if epoch <= self.epoch_s1:
                for batch_data in dl_train_src:
                    losses = self.core_model.batch_fit(batch=batch_data, epoch=epoch)
                    losses['stage'] = 's1'
                    running_losses.append(losses)
            else:
                for batch_data in dl_train_tgt:
                    losses = self.core_model.batch_fit(batch=batch_data, epoch=epoch)
                    losses['stage'] = 's2'
                    running_losses.append(losses)
            show_loss = pd.DataFrame(running_losses).mean().to_dict()
            print(f"[epoch: {epoch:03d}/{self.epochs}, %s" % (
                ', '.join([f'{loss}: {value:.5f}' for loss, value in show_loss.items()])))
            for loss_name, loss in show_loss.items():
                writer.add_scalar(tag=f'loss_train/{loss_name}', scalar_value=loss, global_step=epoch)
            running_losses = pd.DataFrame(running_losses).mean().to_dict()
            self.core_model.after_one_epoch(writer=writer, epoch=epoch, losses=running_losses, metrics={})
            if self.cur_epoch % self.save_epoch_freq == 0:
                self.save_model(model_dir=model_dir, epoch=self.cur_epoch)
            print('\t Epoch: %03d Time Taken: %d sec' % (self.cur_epoch, time.time() - epoch_start_time))
        writer.close()

    def model_train_dev(self, train_dataset: Dataset, dev_dataset: Dataset,
                        model_dir: str, epoch_dev_eval: bool = False, **kwargs):
        log_dir = os.path.join(model_dir, 'log')
        if self.cur_epoch == 0: shutil.rmtree(log_dir, ignore_errors=True)
        writer = SummaryWriter(log_dir, flush_secs=60)
        num_workers = kwargs.get('num_workers', 8)
        dt_train_src, dt_train_tgt = train_dataset
        dl_train_src = DataLoader(dt_train_src, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        dl_train_tgt = DataLoader(dt_train_tgt, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        dev_eval_dl = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.core_model.train()
        for epoch in range(self.cur_epoch + 1, self.epoch_s1 + self.epoch_s2 + 1):
            epoch_start_time = time.time()
            self.cur_epoch = epoch
            running_losses = []
            if epoch <= self.epoch_s1:
                for batch_data in dl_train_src:
                    losses = self.core_model.batch_fit(batch=batch_data, epoch=epoch)
                    running_losses.append(losses)
            else:
                for batch_data in dl_train_tgt:
                    losses = self.core_model.batch_fit(batch=batch_data, epoch=epoch)
                    running_losses.append(losses)
            show_loss = pd.DataFrame(running_losses).mean().to_dict()
            print(f"[epoch: {epoch:03d}/{self.epochs}, %s" % (
                ', '.join([f'{loss}: {value:.5f}' for loss, value in show_loss.items()])))
            for loss_name, loss in show_loss.items():
                writer.add_scalar(tag=f'loss_train/{loss_name}', scalar_value=loss, global_step=epoch)
            running_losses = pd.DataFrame(running_losses).mean().to_dict()
            if epoch_dev_eval and self.cur_epoch % self.eval_epoch_freq == 0:
                results_dev = []
                self.core_model.eval()
                with torch.no_grad():
                    for batch_data in dev_eval_dl:
                        batch_result = self.core_model.batch_predict(batch_data)
                        results_dev.append(batch_result)
                self.core_model.train()
                results_dev = pd.DataFrame(results_dev).to_dict(orient='list')
                results_dev = {k: np.concatenate(v) for k, v in results_dev.items()}
                metrics_dev = yield results_dev
                print(f"\t dev performance of epoch {epoch}: {[f'{k}:{v:.3f}' for k, v in metrics_dev.items()]}")
                writer.add_hparams(hparam_dict={'set': 'val'},
                                   metric_dict={'metric/' + k: v for k, v in metrics_dev.items()},
                                   name='metric_val', global_step=epoch)
            else:
                metrics_dev = {}
            self.core_model.after_one_epoch(writer=writer, epoch=epoch, losses=running_losses, metrics=metrics_dev)
            if self.cur_epoch % self.save_epoch_freq == 0:
                self.save_model(model_dir=model_dir, epoch=self.cur_epoch)
            print('\t Epoch: %03d Time Taken: %d sec' % (self.cur_epoch, time.time() - epoch_start_time))
        writer.close()
