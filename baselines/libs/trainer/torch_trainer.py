import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .trainer import _Trainer
from ..configurer import Configurer
from ..core_model import _TorchCoreModel


class TorchTrainer(_Trainer):
    def __init__(self, cfg: Configurer, core_model: _TorchCoreModel):
        super().__init__(cfg, core_model)
        self.core_model = core_model
        cfg_hparam = cfg.HPARAM
        self.epochs = cfg_hparam.epochs
        self.batch_size = cfg_hparam.batch_size
        self.lr = cfg_hparam.lr
        self.batch_size = cfg_hparam.batch_size
        self.epochs = cfg_hparam.epochs
        self.lr_decay = cfg_hparam.get('lr_decay', None)
        cfg_train = cfg.TRAIN
        self.loss_sample_points = cfg_train.loss_sample_points
        self.eval_epoch_freq = cfg_train.get('eval_epoch_freq', 10)
        self.save_epoch_freq = cfg_train.save_epoch_freq
        self.cur_epoch = 0

    def reset_model(self):
        self.cur_epoch = 0
        self.core_model.init_core_model()
        self.core_model.init_attr_dict()

    def save_model(self, model_dir: str, is_best: bool = False, epoch: int = None):
        super().save_model(model_dir, is_best, epoch)
        core_state_dict = self.core_model.dump_core_model()
        trainer_state_dict = {'cur_epoch': self.cur_epoch}
        all_state_dict = {'core_model': core_state_dict, 'trainer': trainer_state_dict}
        cp_prefix = 'checkpoint'
        if is_best:
            cp_path = os.path.join(model_dir, f'{cp_prefix}-best.pth')
        elif epoch:
            cp_path = os.path.join(model_dir, f"{cp_prefix}-{epoch:03d}.pth")
        else:
            cp_path = os.path.join(model_dir, f'{cp_prefix}-last.pth')
        if os.path.exists(cp_path):
            os.remove(cp_path)
        torch.save(all_state_dict, cp_path)
        print(self.__class__.__name__, ' checkpoint saved: ', cp_path)

    def load_model(self, model_dir: str, is_best: bool = False, epoch: int = None):
        super().load_model(model_dir, is_best, epoch)
        cp_prefix = 'checkpoint'
        if is_best:
            cp_path = os.path.join(model_dir, f'{cp_prefix}-best.pth')
        elif epoch:
            cp_path = os.path.join(model_dir, f"{cp_prefix}-{epoch:03d}.pth")
        else:
            cp_path = os.path.join(model_dir, f'{cp_prefix}-last.pth')
        all_state_dict = torch.load(cp_path)
        core_state_dict = all_state_dict['core_model']
        self.core_model.load_core_model(state_dict=core_state_dict)
        trainer_state_dict = all_state_dict['trainer']
        for attr, value in trainer_state_dict.items():
            if isinstance(value, (int, float, list, dict, tuple)):
                setattr(self, attr, value)
            else:
                raise NotImplementedError
        print(self.__class__.__name__, ' checkpoint loaded: ', cp_path)

    def model_train(self, train_dataset: Dataset, model_dir: str, **kwargs):
        log_dir = os.path.join(model_dir, 'log')
        if self.cur_epoch == 0: shutil.rmtree(log_dir, ignore_errors=True)
        writer = SummaryWriter(log_dir, flush_secs=60)
        num_workers = kwargs.get('num_workers', 8)
        train_dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
        len_dataset = len(train_dataset)
        len_steps = np.ceil(len_dataset / self.batch_size)
        step_line = np.linspace(0, len_steps, num=self.loss_sample_points + 1, dtype=np.int)[1:]
        self.core_model.train()
        for epoch in range(self.cur_epoch + 1, self.epochs + 1):
            epoch_start_time = time.time()
            self.cur_epoch = epoch
            running_losses = []
            for batch_data in train_dl:
                losses = self.core_model.batch_fit(batch=batch_data, epoch=epoch)
                running_losses.append(losses)
                step = len(running_losses)
                if step in step_line:
                    show_loss = pd.DataFrame(running_losses).mean().to_dict()
                    print(f"[epoch: {epoch:03d}/{self.epochs}, step: {step}/{int(len_steps)}, %s]" % (
                        ', '.join([f'{loss}: {value:.5f}' for loss, value in show_loss.items()])))
                    if step == len_steps:
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
        train_dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
        dev_eval_dl = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        len_dataset = len(train_dataset)
        len_steps = np.ceil(len_dataset / self.batch_size)
        step_line = np.linspace(0, len_steps, num=self.loss_sample_points + 1, dtype=np.int)[1:]
        self.core_model.train()
        for epoch in range(self.cur_epoch + 1, self.epochs + 1):
            epoch_start_time = time.time()
            self.cur_epoch = epoch
            running_losses = []
            for batch_data in train_dl:
                losses = self.core_model.batch_fit(batch=batch_data, epoch=epoch)
                running_losses.append(losses)
                step = len(running_losses)
                if step in step_line:
                    show_loss = pd.DataFrame(running_losses).mean().to_dict()
                    print(f"[epoch: {epoch:03d}/{self.epochs}, step: {step}/{int(len_steps)}, %s" % (
                        ', '.join([f'{loss}: {value:.5f}' for loss, value in show_loss.items()])))
                    if step == len_steps:
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

    def model_eval(self, eval_dataset: Dataset, eval_set: str, model_dir: str, **kwargs):
        log_dir = os.path.join(model_dir, 'log')
        if self.cur_epoch == 0: shutil.rmtree(log_dir, ignore_errors=True)
        writer = SummaryWriter(log_dir, flush_secs=60)
        batch_size = kwargs.get('batch_size', self.batch_size)
        num_workers = kwargs.get('num_workers', 4)
        dl = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        results = []
        self.core_model.eval()
        for batch_data in tqdm(dl, desc='forward inference', total=len(dl), dynamic_ncols=True):
            with torch.no_grad():
                batch_result = self.core_model.batch_predict(batch_data)
                results.append(batch_result)
        results = pd.DataFrame(results).to_dict(orient='list')
        results = {k: np.concatenate(v) for k, v in results.items()}
        try:
            images = results['image']
            ch1, ch2, ch3 = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]
            writer.add_histogram(tag=f'hist_{eval_set}/ch1', values=ch1)
            writer.add_histogram(tag=f'hist_{eval_set}/ch2', values=ch2)
            writer.add_histogram(tag=f'hist_{eval_set}/ch3', values=ch3)
        except:
            pass
        writer.close()
        return results
