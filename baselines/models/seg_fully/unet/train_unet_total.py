import os
import sys
import time
from argparse import ArgumentParser

import torch.multiprocessing

sys.path.append('/home/nico/PycharmProjects/defect-detection')
torch.multiprocessing.set_sharing_strategy('file_system')
from baselines.libs.configurer import Configurer
from baselines.libs.processor import ScoreMapProcessor
from baselines.libs.evaluator import ZLEvaluator
from baselines.libs.visualizer import ScoreMapVisualizer
from baselines.libs.trainer import TorchTrainer
from baselines.libs.model_manager import DefectModelManager
from baselines.models.seg_fully.unet.unet import UNet
from baselines.models.seg_fully.misc.mask_dsb import MaskDSB
from baselines.libs.utils import CSVHelper

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cfg', required=True, dest='config_path', type=str, help='the name of configuration file')
    parser.add_argument('--mode', required=True, type=str, help='train/train_dev/test/train_k_fold/test_k_fold')
    parser.add_argument('--cuda', default='3', help='CUDA device ID', type=str)
    parser.add_argument('--date', default=None, type=str, help='please specify the running date when test one model')
    parser.add_argument('--nocsv', action='store_true', help="don't store a csv file to the desktop")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    csv_date = time.strftime("%F", time.localtime())
    csv_helper = CSVHelper(save_dir='~/Desktop/ZL-baseline/', csv_date=csv_date, train_mode=args.mode, append_mode=True)
    cfg = Configurer(args.config_path, verbose=True)
    train_metrics_list = []
    dev_metrics_list = []
    test_metrics_list = []
    assert cfg.DATASET.fabrics == 'total'
    assert args.mode in ['train', 'train_k_fold']
    cfg.DATASET.fabric = cfg.DATASET.fabrics
    print('{0:%^64}'.format(cfg.DATASET.fabric))
    model = UNet(cfg=cfg)
    trainer = TorchTrainer(cfg=cfg, core_model=model)
    ds_builder = MaskDSB(cfg=cfg)
    processor = ScoreMapProcessor(cfg=cfg)
    evaluator = ZLEvaluator(cfg=cfg)
    visualizer = ScoreMapVisualizer(cfg=cfg)
    dmm = DefectModelManager(cfg=cfg, ds_builder=ds_builder, trainer=trainer, processor=processor,
                             evaluator=evaluator, visualizer=visualizer, run_date=args.date)
    train_metrics, dev_metrics, test_metrics = dmm.train_by_mode(train_mode=args.mode, test_after_train=False)
    train_metrics_list.extend(train_metrics)
    dev_metrics_list.extend(dev_metrics)
    test_metrics_list.extend(test_metrics)
