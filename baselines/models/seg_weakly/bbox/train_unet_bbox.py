import os
import sys
import time
from argparse import ArgumentParser

import torch
from torchvision import transforms

sys.path.append('/home/nico/PycharmProjects/defect-detection')
from baselines.libs.processor import ScoreMapProcessor
from baselines.libs.evaluator import ZLEvaluator
from baselines.libs.visualizer import ScoreMapVisualizer
from baselines.libs.model_manager import DefectModelManager
from baselines.libs.trainer import TorchTrainer
from baselines.libs.utils import CSVHelper
from baselines.models.seg_fully.misc.mask_dsb import MaskDSB
from baselines.libs.configurer import Configurer
from baselines.libs.dataset import _ObjectDataset
from baselines.models.seg_weakly.bbox.unet_bbox import UNetBBox


class EstMaskDataset(_ObjectDataset):
    def __init__(self, zl_imgs, normalization, color_mode, img_size=None) -> None:
        augment_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ])
        super().__init__(zl_imgs, normalization, color_mode, img_size, transform=augment_transform)

    def __getitem__(self, index):
        zl_image = self.zl_imgs[index]
        name = zl_image.id
        label = zl_image.info()['defective']
        img = zl_image.image()
        bboxes = zl_image.annotation()
        w_s, h_s = img.size
        img = transforms.Compose([
            transforms.Resize(size=self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])(img)
        h_t, w_t = img.shape[1:3]
        h_scale, w_scale = h_t / h_s, w_t / w_s
        est_mask = torch.zeros([h_t, w_t], dtype=torch.float32)
        for bbox in bboxes:
            y1, y2, x1, x2 = bbox
            y1 = int(float(y1) * h_scale)
            y2 = int(float(y2) * h_scale)
            x1 = int(float(x1) * w_scale)
            x2 = int(float(x2) * w_scale)
            est_mask[y1:y2, x1:x2] = 1
        est_mask = est_mask.unsqueeze(0)
        return name, label, img, est_mask,


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
        model = UNetBBox(cfg=cfg)
        trainer = TorchTrainer(cfg=cfg, core_model=model)
        ds_builder = MaskDSB(cfg=cfg, TrainDataset=EstMaskDataset)
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
