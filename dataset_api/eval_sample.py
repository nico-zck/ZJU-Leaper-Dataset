# -*- coding: utf-8 -*-
"""
@Time   : 2020/11/19 5:51 下午
@Author : Nico
"""

import numpy as np
import pandas as pd

from .eval_pixel import _compute_confusion_matrix, _f1_score, _acc, _fpr

# the size constrain is according to pixel ratio
MIN_AREA_RATIO = 1. / 128.


def evaluation_sample(binary_pixel_pred, binary_pixel_target, info_region):
    info_pred: pd.DataFrame = info_region["info_pred"]  # columns: id_img, id_pred, bbox
    info_target: pd.DataFrame = info_region["info_target"]  # columns: id_img, id_target, bbox
    # columns=['id_pred', 'id_target', 'area_pred', 'area_target', 'area_union', 'area_overlap', 'ratio_overlap', 'iou', 'id_img']
    info_overlap: pd.DataFrame = info_region["info_overlap"]

    num_img = len(binary_pixel_target)
    _, h, w = binary_pixel_target.squeeze().shape
    MIN_AREA_TH = (np.float_(h) * MIN_AREA_RATIO) * (np.float_(w) * MIN_AREA_RATIO)

    label_pred = np.zeros(num_img, dtype=np.bool)
    if not info_pred.empty:
        img_pred_ids = info_pred[info_pred['area_pred'] > MIN_AREA_TH]['id_img'].unique()
        label_pred[img_pred_ids] = 1

    label_target = np.zeros(num_img, dtype=np.bool)
    if not info_target.empty:
        img_target_ids = info_target['id_img'].unique()
        label_target[img_target_ids] = 1

    # tn, fp, fn, tp
    confusion = _compute_confusion_matrix(binary_target=label_target, binary_pred=label_pred)
    precision, recall, f1_score = _f1_score(confusion)

    sample_metrics = dict(
        Pre=precision,
        Rec=recall,
        F1=f1_score,
        Acc=_acc(confusion),
        FPR=_fpr(confusion),
    )

    # metrics_list = []
    # for info_valid in info_valid_list:
    #     iou_threshold = info_valid["iou"]
    #     # columns: ['id_pred', 'area_pred', 'id_img', 'id_target', 'area_target', 'area_union', 'area_overlap', 'ratio_overlap', 'iou']
    #     info_pred_valid: pd.DataFrame = info_valid["info_pred_valid"]
    #     # columns: id_target, area_target, area_pred, area_overlap, id_img
    #     info_target_valid: pd.DataFrame = info_valid["info_target_valid"]
    #     TP = TN = FP = FN = 0
    #     for img_id in range(num_img):
    #         target_img = info_target[info_target["id_img"] == img_id]
    #         if target_img.empty:
    #             pred_img = info_pred[info_pred["id_img"] == img_id]
    #             if pred_img.empty:
    #                 TN += 1
    #             else:
    #                 FP += 1
    #         else:
    #             target_img_valid = info_target_valid[info_target_valid["id_img"] == img_id]
    #             if target_img_valid.empty:
    #                 pred_img = info_pred[info_pred["id_img"] == img_id]
    #                 if pred_img.empty:
    #                     FN += 1
    #                 else:
    #                     FP += 1
    #             else:
    #                 TP += 1
    #     assert TP + FP + TN + FN == num_img
    #     confusion = (TN, FP, FN, TP)
    #     precision, recall, f1_score = _f1_score(confusion)
    #     metrics_list.append(dict(
    #         # iou=iou_threshold,
    #         Pre=precision,
    #         Rec=recall,
    #         F1=f1_score,
    #         Acc=_acc(confusion),
    #         FPR=_fpr(confusion),
    #     ))
    # metrics_list = pd.DataFrame(metrics_list, columns=list(metrics_list[0].keys()))
    # sample_metrics = metrics_list.mean()

    return sample_metrics
