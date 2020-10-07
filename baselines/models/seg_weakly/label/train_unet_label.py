import os
import sys
import time
from argparse import ArgumentParser

from torchvision import transforms

sys.path.append('/home/nico/PycharmProjects/defect-detection')
from baselines.libs.configurer import Configurer
from baselines.libs.dataset.base_datasets import _LabelDataset
from baselines.models.seg_weakly.label.unet_label import UNetLabel
from baselines.libs.processor import ScoreMapProcessor
from baselines.libs.visualizer import ScoreMapVisualizer
from baselines.libs.evaluator import ZLEvaluator
from baselines.libs.model_manager import DefectModelManager
from baselines.libs.trainer import TorchTrainer
from baselines.models.seg_fully.misc.mask_dsb import MaskDSB
from baselines.libs.utils import CSVHelper


class LabelAugDataset(_LabelDataset):
    def __init__(self, zl_imgs, normalization, color_mode, img_size=None) -> None:
        augment_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=180, translate=[0.2, 0.2], scale=[0.8, 1.2], shear=20),
        ])
        super().__init__(zl_imgs=zl_imgs, normalization=normalization, color_mode=color_mode, img_size=img_size,
                         transform=augment_transform)


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
    if cfg.DATASET.fabrics == 'all': cfg.DATASET.fabrics = range(1, 20)
    for p_id in cfg.DATASET.fabrics:
        cfg.DATASET.fabric = 'pattern{0}'.format(p_id)
        print('{0:%^64}'.format(cfg.DATASET.fabric))
        if p_id > 15:
            cfg.DATASET.color_mode = 'gray'
        model = UNetLabel(cfg=cfg)
        trainer = TorchTrainer(cfg=cfg, core_model=model)
        ds_builder = MaskDSB(cfg=cfg, TrainDataset=LabelAugDataset)
        processor = ScoreMapProcessor(cfg=cfg)
        evaluator = ZLEvaluator(cfg=cfg)
        visualizer = ScoreMapVisualizer(cfg=cfg)
        dmm = DefectModelManager(cfg=cfg, ds_builder=ds_builder, trainer=trainer, processor=processor,
                                 evaluator=evaluator, visualizer=visualizer, run_date=args.date)
        if p_id <= 0:
            train_metrics, dev_metrics, test_metrics = dmm.train_by_mode(train_mode=args.mode, resume=True)
        else:
            train_metrics, dev_metrics, test_metrics = dmm.train_by_mode(train_mode=args.mode)
        train_metrics_list.extend(train_metrics)
        dev_metrics_list.extend(dev_metrics)
        test_metrics_list.extend(test_metrics)
        if not args.nocsv:
            csv_helper.save(model_name=dmm.general_name,
                            metrics_lists=[train_metrics_list, dev_metrics_list, test_metrics_list])
