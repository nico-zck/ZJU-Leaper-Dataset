# -*- coding: utf-8 -*-
"""
@Time   : 2019/11/27 5:03 下午
@Author : Nico
"""
import itertools
import json
import os
import random
import warnings
import xml.etree.cElementTree as ET
from typing import Union, List, Tuple

import numpy as np
from PIL import Image

CONFIG = {
    'setting1': {'normal_train': 'train', 'defect_train': 'none', 'ann_train': 'none',
                 'normal_test': 'test', 'defect_test': 'test', 'ann_test': 'mask',
                 'normal_train_eval': 'train', 'defect_train_eval': 'none', 'ann_train_eval': 'mask',
                 'ann_eval': 'mask'},

    'setting2': {'normal_train': 'train', 'defect_train': 'small', 'ann_train': 'mask',
                 'normal_test': 'test', 'defect_test': 'test', 'ann_test': 'mask',
                 'normal_train_eval': 'train', 'defect_train_eval': 'small', 'ann_train_eval': 'mask',
                 'ann_eval': 'mask'},

    'setting3': {'normal_train': 'train', 'defect_train': 'train', 'ann_train': 'label',
                 'normal_test': 'test', 'defect_test': 'test', 'ann_test': 'mask',
                 'normal_train_eval': 'train', 'defect_train_eval': 'train', 'ann_train_eval': 'mask',
                 'ann_eval': 'mask'},

    'setting4': {'normal_train': 'train', 'defect_train': 'train', 'ann_train': 'bbox',
                 'normal_test': 'test', 'defect_test': 'test', 'ann_test': 'mask',
                 'normal_train_eval': 'train', 'defect_train_eval': 'train', 'ann_train_eval': 'mask',
                 'ann_eval': 'mask'},

    'setting5': {'normal_train': 'train', 'defect_train': 'train', 'ann_train': 'mask',
                 'normal_test': 'test', 'defect_test': 'test', 'ann_test': 'mask',
                 'normal_train_eval': 'train', 'defect_train_eval': 'train', 'ann_train_eval': 'mask',
                 'ann_eval': 'mask'},
}


def chunks(seq, n):
    """
    split a list into N chunks
    :param seq:
    :param n:
    :return:
    """
    assert len(seq) > n
    avg = len(seq) / float(n)
    out = []
    last = 0
    while round(last) < len(seq):
        out.append(seq[round(last):round(last + avg)])
        last += avg
    return out


class ZLImage:
    """
    Packaging all meta data and functions of a image into one object.
    """

    def __init__(self, img_id: int, ann_type: str, dataset_dir: str):
        self.id = img_id
        self.ann = ann_type

        if os.path.exists('%s/Images/000001.jpg' % dataset_dir):
            self.img_path = '%s/Images/{id}.jpg' % dataset_dir
        elif os.path.exists('%s/Images/000001.png' % dataset_dir):
            self.img_path = '%s/Images/{id}.png' % dataset_dir
        else:
            if not os.path.exists('%s/Images'):
                raise NotImplementedError("Images folder does not exist!")
            else:
                raise NotImplementedError("not supported image format or wrong length of image id!")

        self.xml_path = '%s/Annotations/xmls/{id}.xml' % dataset_dir
        self.mask_path = '%s/Annotations/masks/{id}.png' % dataset_dir

    def image(self) -> Image:
        img_path = self.img_path.format(id=self.id)
        img = Image.open(img_path)
        return img

    def info(self) -> dict:
        """
        Return the label info of fabric image.
        :return: {'pattern': int, 'id': int, 'defective': boolean}
        """
        xml_path = self.xml_path.format(id=self.id)
        p_id = int(ET.parse(xml_path).find('pattern').text)
        defect_flag = bool(int(ET.parse(xml_path).find('defective').text))
        info = {'pattern': p_id, 'id': self.id, 'defective': defect_flag}
        return info

    def annotation(self, ann_type: str = None):
        """
        Get annotation of image from ZJU-Leaper dataset.
        Set ann_type="mask" when using images for evaluating.
        :param ann_type:
        :return:
        """
        if ann_type is None: ann_type = self.ann
        if ann_type != self.ann:
            warnings.warn('Please note that the annotation type is mismatch with the dataset setting!')

        if ann_type == 'label':
            xml_path = self.xml_path.format(id=self.id)
            ann = int(ET.parse(xml_path).find('defective').text)
        elif ann_type == 'bbox':
            xml_path = self.xml_path.format(id=self.id)
            objs = ET.parse(xml_path).findall('bbox')
            ann = []
            for ix, bbox in enumerate(objs):
                y1 = int(float(bbox.find('ymin').text))
                y2 = int(float(bbox.find('ymax').text))
                x1 = int(float(bbox.find('xmin').text))
                x2 = int(float(bbox.find('xmax').text))
                ann.append((y1, y2, x1, x2))
        elif ann_type == 'mask':
            mask_path = self.mask_path.format(id=self.id)
            if os.path.exists(mask_path):
                ann = Image.open(mask_path).convert('L')
            else:
                ann = Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert('L')
        elif ann_type == 'none':
            ann = []
        else:
            raise NotImplementedError
        return ann


# define typing hint
ZLIMGS = List[ZLImage]


class ZLFabric:
    def __init__(self, dir: str, fabric: Union[str, int], setting: Union[str, int], seed: int = None):
        """
        Create an object to manage ZJU-Leaper dataset.

        :param dir: the base directory for the ZJU-Leaper dataset.
        :param fabric: int or string to specify using "patternX" or "groupX".
        :param setting:
            |Available strings for setting parameter:
            |   "setting1": Normal sample only (annotation-free);
            |   "setting2": Small amount of defect data (with mask annotation);
            |   "setting3": Large amount of defect data (with label annotation);
            |   "setting4": Large amount of defect data (with bounding-box annotation);
            |   "setting5": Large amount of defect data (with mask annotation);
        :param seed: random seed.
        """
        self.dataset_dir = dir

        if isinstance(fabric, int):
            fabric = 'pattern%d' % fabric
        assert fabric in ['pattern%d' % i for i in range(1, 20)] \
               or fabric in ['group%d' % i for i in range(1, 6)] or fabric == 'total'
        self.fabric = fabric

        if isinstance(setting, int):
            setting = 'setting%d' % setting
        assert setting in ['setting%d' % i for i in range(1, 6)]
        self.setting = setting

        if 'pattern' in self.fabric:
            self.json_path = '%s/ImageSets/Patterns/%s.json' % (dir, fabric)
        elif 'group' in self.fabric:
            self.json_path = '%s/ImageSets/Groups/%s.json' % (dir, fabric)
        elif 'total' == self.fabric:
            self.json_path = '%s/ImageSets/total.json' % (dir)
        else:
            raise NotImplementedError

        self.rnd = random.Random(seed)

    def create_zl_imgs_given_ids(self, ids: list, subset: str, ann_type: str) -> ZLIMGS:
        """

        :param ids:
        :param subset: ["none", "small", "train", "dev", "test"]
        :param ann_type: ["none", "label", "bbox", "mask"]
        :return:
        """
        assert subset in ["none", "small", "train", "dev", "test"]
        assert ann_type in ["none", "label", "bbox", "mask"]

        if subset == 'none':
            ids = []
        elif subset == 'small':
            ids = self.rnd.sample(ids, len(ids) // 10)
        else:
            pass

        zl_imgs = []
        for id in ids:
            zl_img = ZLImage(img_id=id, ann_type=ann_type, dataset_dir=self.dataset_dir)
            zl_imgs.append(zl_img)

        return zl_imgs

    def prepare_train(self) -> Tuple[ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS]:
        """
        Preparing ZLImages from training set.
        :return: [Normal ZLImages, Defective ZLImages]
        """

        if self.setting == 'setting1':
            warnings.warn("Please note that Setting 1 should not use train eval dataset! "
                          "Because its training set only contain normal samples!")

        with open(self.json_path) as fp:
            ids_json = json.load(fp)
        ids_train_normal = ids_json['normal']['train']
        ids_train_defect = ids_json['defect']['train']

        # train
        zlimgs_train_normal = self.create_zl_imgs_given_ids(ids=ids_train_normal,
                                                            subset=CONFIG[self.setting]['normal_train'],
                                                            ann_type=CONFIG[self.setting]['ann_train'])
        zlimgs_train_defect = self.create_zl_imgs_given_ids(ids=ids_train_defect,
                                                            subset=CONFIG[self.setting]['defect_train'],
                                                            ann_type=CONFIG[self.setting]['ann_train'])

        # train eval
        zlimgs_train_eval_normal = self.create_zl_imgs_given_ids(ids=ids_train_normal,
                                                                 subset=CONFIG[self.setting]['normal_train'],
                                                                 ann_type=CONFIG[self.setting]['ann_eval'])
        zlimgs_train_eval_defect = self.create_zl_imgs_given_ids(ids=ids_train_defect,
                                                                 subset=CONFIG[self.setting]['defect_train'],
                                                                 ann_type=CONFIG[self.setting]['ann_eval'])

        return zlimgs_train_normal, zlimgs_train_defect, zlimgs_train_eval_normal, zlimgs_train_eval_defect

    def prepare_k_fold(self, k_fold: int, shuffle: bool = True) \
            -> List[Tuple[ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS, ZLIMGS]]:
        """
        Preparing ZLImages from cross validation.
        :return: [(zlimgs_k_train_normal, zlimgs_k_train_defect, zlimgs_k_dev_normal, zlimgs_k_dev_defect)]
        """
        if self.setting == 'setting1':
            warnings.warn("Setting 1 should not use train eval dataset! "
                          "Because its training set only contain normal samples!")

        with open(self.json_path) as fp:
            ids_json = json.load(fp)
        ids_train_normal = ids_json['normal']['train']
        ids_train_defect = ids_json['defect']['train']

        if shuffle:
            self.rnd.shuffle(ids_train_normal)
            self.rnd.shuffle(ids_train_defect)
        ids_folds_normal = chunks(ids_train_normal, k_fold)
        ids_folds_defect = chunks(ids_train_defect, k_fold)

        zlimgs_folds = []
        for k in range(k_fold):
            _ids_folds_normal = ids_folds_normal.copy()
            _ids_folds_defect = ids_folds_defect.copy()
            ids_k_dev_normal = _ids_folds_normal.pop(k)
            ids_k_dev_defect = _ids_folds_defect.pop(k)
            ids_k_train_normal = list(itertools.chain(*_ids_folds_normal))
            ids_k_train_defect = list(itertools.chain(*_ids_folds_defect))

            # train
            zlimgs_k_train_normal = self.create_zl_imgs_given_ids(ids=ids_k_train_normal,
                                                                  subset=CONFIG[self.setting]['normal_train'],
                                                                  ann_type=CONFIG[self.setting]['ann_train'])
            zlimgs_k_train_defect = self.create_zl_imgs_given_ids(ids=ids_k_train_defect,
                                                                  subset=CONFIG[self.setting]['defect_train'],
                                                                  ann_type=CONFIG[self.setting]['ann_train'])
            # train-eval
            zlimgs_k_train_eval_normal = self.create_zl_imgs_given_ids(ids=ids_k_train_normal,
                                                                       subset=CONFIG[self.setting]['normal_train'],
                                                                       ann_type=CONFIG[self.setting]['ann_eval'])
            zlimgs_k_train_eval_defect = self.create_zl_imgs_given_ids(ids=ids_k_train_defect,
                                                                       subset=CONFIG[self.setting]['defect_train'],
                                                                       ann_type=CONFIG[self.setting]['ann_eval'])
            # dev
            zlimgs_k_dev_normal = self.create_zl_imgs_given_ids(ids=ids_k_dev_normal, subset='dev',
                                                                ann_type=CONFIG[self.setting]['ann_eval'])
            zlimgs_k_dev_defect = self.create_zl_imgs_given_ids(ids=ids_k_dev_defect, subset='dev',
                                                                ann_type=CONFIG[self.setting]['ann_eval'])
            zlimgs_folds.append((zlimgs_k_train_normal, zlimgs_k_train_defect,
                                 zlimgs_k_train_eval_normal, zlimgs_k_train_eval_defect,
                                 zlimgs_k_dev_normal, zlimgs_k_dev_defect))
        return zlimgs_folds

    def prepare_test(self) -> Tuple[ZLIMGS, ZLIMGS]:
        """
        Preparing ZLImages from test set.
        :return: [Normal ZLImages, Defective ZLImages]
        """
        with open(self.json_path) as fp:
            ids_json = json.load(fp)
        ids_test_normal = ids_json['normal']['test']
        ids_test_defect = ids_json['defect']['test']

        # test
        zlimgs_test_normal = self.create_zl_imgs_given_ids(ids=ids_test_normal,
                                                           subset=CONFIG[self.setting]['normal_test'],
                                                           ann_type=CONFIG[self.setting]['ann_test'])
        zlimgs_test_defect = self.create_zl_imgs_given_ids(ids=ids_test_defect,
                                                           subset=CONFIG[self.setting]['defect_test'],
                                                           ann_type=CONFIG[self.setting]['ann_test'])
        return zlimgs_test_normal, zlimgs_test_defect
