import time
from datetime import datetime
from multiprocessing import Process
from os import path, makedirs, getpid
from pathlib import Path
from typing import Tuple, List, Dict

import psutil
from tensorboardX import SummaryWriter

from .configurer import Configurer
from .dataset.ds_builder import _DSBuilder
from .evaluator.evaluator import _Evaluator
from .processor.processor import _Processor
from .trainer.trainer import _Trainer
from .utils import join_str_list
from .visualizer.visualizer import _Visualizer


class DefectModelManager(object):
    def __init__(self, cfg: Configurer,
                 ds_builder: _DSBuilder,
                 trainer: _Trainer,
                 processor: _Processor,
                 evaluator: _Evaluator,
                 visualizer: _Visualizer = None,
                 run_date=None):
        self.cfg = cfg
        self.ds_builder = ds_builder
        self.trainer = trainer
        self.processor = processor
        self.evaluator = evaluator
        self.visualizer = visualizer
        if run_date:
            try:
                datetime.strptime(run_date, "%y%m%d")
            except ValueError:
                raise ValueError('date string has wrong format! please use the format like "190101".')
        else:
            run_date = datetime.fromtimestamp(psutil.Process(getpid()).create_time()).strftime("%y%m%d")
        self.run_date = run_date
        self._init_model_folder_and_name()
        cfg.print_configuration()

    def _init_model_folder_and_name(self):
        trainer_name = self.trainer.name
        ds_name = self.ds_builder.dataset_name
        dsb_name = self.ds_builder.name
        cfg_name = self.cfg.name
        if hasattr(self.cfg, "NOTE"):
            notes = '-'.join([f"{k}_{v}" for k, v in self.cfg.NOTE.items()])
        else:
            notes = None
        self.base_dir = join_str_list(trainer_name, self.run_date, delimiter='-')
        self.short_name = join_str_list(dsb_name, cfg_name, notes, delimiter='-')
        self.model_dir = path.join(Path(__file__).parents[1], 'saved_models/', self.base_dir, self.short_name)
        self.full_name = self.base_dir + '__' + self.short_name
        self.general_name = join_str_list(ds_name, cfg_name, notes, delimiter='-')
        self.general_name = self.base_dir + '__' + self.general_name
        self.cfg.TEMP.model_dir = self.model_dir
        self.cfg.TEMP.short_name = self.short_name
        self.cfg.TEMP.full_name = self.full_name
        self.cfg.TEMP.general_name = self.general_name

    def train(self, eval_train: bool, resume: bool = False, **kwargs) -> dict:
        makedirs(self.model_dir, exist_ok=True)
        print(self.__class__.__name__, 'model dir: ', self.model_dir)
        print('{0:*^64}'.format('start training'))
        dt_train, dt_train_eval = self.ds_builder.build_train()
        print(f"length of training dataset (all): {len(dt_train)}")
        print(f"length of evaluation dataset (train): {len(dt_train_eval)}")
        if resume:
            resume_epoch = kwargs.get('resume_epoch', None)
            self.trainer.load_model(self.model_dir, epoch=resume_epoch)
        self.trainer.model_train(train_dataset=dt_train, model_dir=self.model_dir, **kwargs)
        self.trainer.save_model(model_dir=self.model_dir)
        print('{0:*^64}'.format('end training'))
        if eval_train == True:
            print('{0:*^64}'.format('training set performance'))
            print(f'length of dataset: {len(dt_train_eval)}')
            start = time.time()
            train_results = self.trainer.model_eval(eval_dataset=dt_train_eval, eval_set='train',
                                                    model_dir=self.model_dir, **kwargs)
            train_results = self.processor.processing(result_dict=train_results, **kwargs)
            train_metrics = self.evaluator.evaluating(model_dir=self.model_dir, result_dict=train_results,
                                                      subset='train', **kwargs)
            end = time.time()
            print('{0:*^64}'.format('taken %d sec' % (end - start)))
            return train_metrics
        else:
            return {}

    def train_dev(self, eval_train: bool, epoch_dev_eval: bool, resume: bool = False, **kwargs) -> Tuple[Dict, Dict]:
        makedirs(self.model_dir, exist_ok=True)
        print(self.__class__.__name__, 'model dir: ', self.model_dir)
        print('{0:*^64}'.format('start training'))
        dt_train, dt_train_eval, dt_dev_eval = self.ds_builder.build_train_dev()
        print(f"length of training dataset (train): {len(dt_train)}")
        print(f"length of evaluation dataset (train): {len(dt_train_eval)}")
        print(f"length of evaluation dataset (dev): {len(dt_dev_eval)}")
        if resume:
            resume_epoch = kwargs.get('resume_epoch', None)
            self.trainer.load_model(self.model_dir, epoch=resume_epoch)
        g = self.trainer.model_train_dev(train_dataset=dt_train, dev_dataset=dt_dev_eval, model_dir=self.model_dir,
                                         epoch_dev_eval=epoch_dev_eval)
        try:
            epoch_results = next(g)
            while True:
                epoch_results = self.processor.epoch_processing(result_dict=epoch_results)
                epoch_metrics = self.evaluator.epoch_evaluating(result_dict=epoch_results)
                epoch_results = g.send(epoch_metrics)
        except StopIteration:
            self.trainer.save_model(model_dir=self.model_dir)
        print('{0:*^64}'.format('end training'))
        if eval_train == True:
            print('{0:*^64}'.format('training set performance'))
            print(f'length of dataset: {len(dt_train_eval)}')
            start = time.time()
            train_results = self.trainer.model_eval(eval_dataset=dt_train_eval, eval_set='train',
                                                    model_dir=self.model_dir, **kwargs)
            train_results = self.processor.processing(result_dict=train_results, **kwargs)
            train_metrics = self.evaluator.evaluating(model_dir=self.model_dir, result_dict=train_results,
                                                      subset='train', **kwargs)
            end = time.time()
            print('{0:*^64}'.format('taken %d sec' % (end - start)))
        else:
            train_metrics = {}
        print('{0:*^64}'.format(f'validation set performance'))
        print(f'length of dataset: {len(dt_dev_eval)}')
        start = time.time()
        dev_results = self.trainer.model_eval(eval_dataset=dt_dev_eval, eval_set='val', model_dir=self.model_dir,
                                              **kwargs)
        dev_results = self.processor.processing(result_dict=dev_results, **kwargs)
        dev_metrics = self.evaluator.evaluating(model_dir=self.model_dir, result_dict=dev_results, subset='val',
                                                **kwargs)
        end = time.time()
        print('{0:*^64}'.format('taken %d sec' % (end - start)))
        return train_metrics, dev_metrics

    def test(self, load_checkpoint: bool, n_fold: int = None,
             figure: bool = True, vis_rate: float = 0.1, vis_bad: bool = False, **kwargs) -> dict:
        if n_fold:
            model_dir = path.join(self.model_dir, f'fold-{n_fold}')
        else:
            model_dir = self.model_dir
        if load_checkpoint:
            self.trainer.load_model(model_dir=model_dir)
        if not path.exists(model_dir):
            raise FileExistsError('model directory is not exists !!')
        print(self.__class__.__name__, 'model dir: ', model_dir)
        print('{0:*^64}'.format('test-set performance'))
        dt_test = self.ds_builder.build_test()
        print(f"length of evaluation dataset (test): {len(dt_test)}")
        start = time.time()
        test_results = self.trainer.model_eval(eval_dataset=dt_test, eval_set='test', model_dir=model_dir, **kwargs)
        test_results = self.processor.processing(result_dict=test_results, **kwargs)
        test_metrics = self.evaluator.evaluating(model_dir=model_dir, result_dict=test_results,
                                                 subset='test', n_fold=n_fold, **kwargs)
        if self.visualizer:
            background_plot = kwargs.get('background_plot', False)
            if background_plot:
                print('visualizer will run in background, and it may mess up the output.')
                Process(target=self.visualizer.visualizing,
                        kwargs=dict(model_dir=model_dir, result_dict=test_results, figure=figure,
                                    vis_rate=vis_rate, vis_bad=vis_bad, kwargs=kwargs)).start()
            else:
                self.visualizer.visualizing(model_dir=model_dir, result_dict=test_results, figure=figure,
                                            vis_rate=vis_rate, vis_bad=vis_bad, **kwargs)
        end = time.time()
        print('{0:*^64}'.format('taken %d sec' % (end - start)))
        try:
            test_metrics['weights'] = str(dict(self.trainer.core_model.model_T.named_parameters()))
            with SummaryWriter(logdir=path.join(model_dir, 'log')) as w:
                w.add_text(tag='weights', text_string=test_metrics['weights'])
        except:
            pass
        try:
            test_metrics['forward'] = str(dict(self.trainer.core_model.forward_T.named_parameters()))
            test_metrics['backward'] = str(dict(self.trainer.core_model.backward_T.named_parameters()))
            with SummaryWriter(logdir=path.join(model_dir, 'log')) as w:
                w.add_text(tag='forward', text_string=test_metrics['forward'])
                w.add_text(tag='backward', text_string=test_metrics['backward'])
        except:
            pass
        return test_metrics

    def train_k_fold(self, eval_train: bool, epoch_dev_eval: bool, resume: bool = False, **kwargs) \
            -> Tuple[List[Dict], List[Dict]]:
        dt_train_list, dt_train_eval_list, dt_dev_eval_list = self.ds_builder.build_k_fold()
        print(f"length of training datasets (k-fold): {[len(t) for t in dt_train_list]}")
        print(f"length of evaluation datasets (k-fold train): {[len(t) for t in dt_train_eval_list]}")
        print(f"length of evaluation datasets (k-fold valid): {[len(t) for t in dt_dev_eval_list]}")
        train_metrics_list = []
        dev_metrics_list = []
        for fold_idx, (dt_train, dt_train_eval, dt_dev_eval) in \
                enumerate(zip(dt_train_list, dt_train_eval_list, dt_dev_eval_list), start=1):
            print('{0:*^64}'.format(f'start training {fold_idx}-fold'))
            fold_idx_dir = path.join(self.model_dir, f'fold-{fold_idx}')
            makedirs(fold_idx_dir, exist_ok=True)
            print(self.__class__.__name__, 'model dir: ', fold_idx_dir)
            if fold_idx > 1: self.trainer.reset_model()
            if resume:
                resume_fold: int = kwargs.get('resume_fold')
                if fold_idx < resume_fold:
                    print(f'skipping the training of fold-{fold_idx}.')
                    continue
                elif fold_idx == resume_fold:
                    if 'resume_epoch' in kwargs:
                        resume_epoch: int = kwargs.get('resume_epoch')
                        self.trainer.load_model(fold_idx_dir, epoch=resume_epoch)
                    else:
                        pass
                else:
                    pass
            g = self.trainer.model_train_dev(train_dataset=dt_train, dev_dataset=dt_dev_eval,
                                             model_dir=fold_idx_dir, epoch_dev_eval=epoch_dev_eval)
            try:
                epoch_results = next(g)
                while True:
                    epoch_results = self.processor.epoch_processing(result_dict=epoch_results)
                    epoch_metrics = self.evaluator.epoch_evaluating(result_dict=epoch_results)
                    epoch_results = g.send(epoch_metrics)
            except (StopIteration, TypeError):
                self.trainer.save_model(model_dir=fold_idx_dir)
                print('{0:*^64}'.format(f'end training {fold_idx}-fold'))
            if eval_train == True:
                print('{0:*^64}'.format(f'{fold_idx}-fold training set performance'))
                print(f'length of dataset: {len(dt_train_eval)}')
                start = time.time()
                train_results = self.trainer.model_eval(eval_dataset=dt_train_eval, eval_set='train',
                                                        model_dir=fold_idx_dir, **kwargs)
                train_results = self.processor.processing(result_dict=train_results, **kwargs)
                train_metrics = self.evaluator.evaluating(model_dir=fold_idx_dir, result_dict=train_results,
                                                          subset='train', n_fold=fold_idx, **kwargs)
                end = time.time()
                print('{0:*^64}'.format('taken %d sec' % (end - start)))
                train_metrics_list.append(train_metrics)
            else:
                train_metrics_list.append({})
            print('{0:*^64}'.format(f'{fold_idx}-fold validation set performance'))
            print(f'length of dataset: {len(dt_dev_eval)}')
            start = time.time()
            dev_results = self.trainer.model_eval(eval_dataset=dt_dev_eval, eval_set='val',
                                                  model_dir=fold_idx_dir, **kwargs)
            dev_results = self.processor.processing(result_dict=dev_results, **kwargs)
            dev_metrics = self.evaluator.evaluating(model_dir=fold_idx_dir, result_dict=dev_results, subset='val',
                                                    n_fold=fold_idx, **kwargs)
            end = time.time()
            print('{0:*^64}'.format('taken %d sec' % (end - start)))
            dev_metrics_list.append(dev_metrics)
        return train_metrics_list, dev_metrics_list

    def test_k_fold(self, folds: list = None, figure=True, vis_rate=0.1, vis_bad=False, **kwargs) -> List[Dict]:
        test_metrics_list = []
        if folds == None:
            folds = range(1, self.cfg.DATASET.k_fold + 1)
        for k in folds:
            self.cfg.TEMP.fold = k
            print('{0:*^70}'.format('n-fold: %d' % k))
            test_metrics = self.test(load_checkpoint=True, n_fold=k,
                                     figure=figure, vis_bad=vis_bad, vis_rate=vis_rate, **kwargs)
            test_metrics_list.append(test_metrics)
        return test_metrics_list

    def train_by_mode(self, train_mode: str, test_after_train: bool = True, resume: bool = False, **kwargs) -> \
            Tuple[List[Dict], List[Dict], List[Dict]]:
        train_cfg = self.cfg.TRAIN
        if train_mode == 'train':
            train_metrics = self.train(eval_train=train_cfg.eval_train, resume=resume, **kwargs)
            dev_metrics = {}
            if test_after_train:
                test_metrics = self.test(load_checkpoint=False,
                                         figure=train_cfg.figure,
                                         vis_rate=train_cfg.vis_rate,
                                         vis_bad=train_cfg.vis_bad)
            else:
                test_metrics = {}
            train_metrics, dev_metrics, test_metrics = [train_metrics], [dev_metrics], [test_metrics]
        elif train_mode == 'train_dev':
            train_metrics, dev_metrics = self.train_dev(eval_train=train_cfg.eval_train,
                                                        epoch_dev_eval=train_cfg.epoch_dev_eval,
                                                        resume=resume, **kwargs)
            if test_after_train:
                test_metrics = self.test(load_checkpoint=False,
                                         figure=train_cfg.figure,
                                         vis_rate=train_cfg.vis_rate,
                                         vis_bad=train_cfg.vis_bad)
            else:
                test_metrics = {}
            train_metrics, dev_metrics, test_metrics = [train_metrics], [dev_metrics], [test_metrics]
        elif train_mode == 'test':
            train_metrics = dev_metrics = {}
            test_metrics = self.test(load_checkpoint=True,
                                     figure=train_cfg.figure,
                                     vis_rate=train_cfg.vis_rate,
                                     vis_bad=train_cfg.vis_bad)
            train_metrics, dev_metrics, test_metrics = [train_metrics], [dev_metrics], [test_metrics]
        elif train_mode == 'train_k_fold':
            train_metrics, dev_metrics = self.train_k_fold(eval_train=train_cfg.eval_train,
                                                           epoch_dev_eval=train_cfg.epoch_dev_eval,
                                                           resume=resume, **kwargs)
            if test_after_train:
                test_metrics = self.test_k_fold(figure=train_cfg.figure,
                                                vis_rate=train_cfg.vis_rate,
                                                vis_bad=train_cfg.vis_bad)
            else:
                test_metrics = []
        elif train_mode == 'test_k_fold':
            folds = train_cfg.get("folds", None)
            train_metrics = dev_metrics = []
            test_metrics = self.test_k_fold(folds=folds,
                                            figure=train_cfg.figure,
                                            vis_rate=train_cfg.vis_rate,
                                            vis_bad=train_cfg.vis_bad)
        else:
            raise NotImplementedError
        return train_metrics, dev_metrics, test_metrics
