# -*- coding: utf-8 -*-
"""
@Time   : 2019-05-07 23:53
@Author : Nico
"""
import numpy as np
from collections import OrderedDict
from typing import Tuple


def _compute_confusion_matrix(binary_target: np.ndarray, binary_pred: np.ndarray):
    tp = np.sum(binary_pred[binary_target == 1] == 1, dtype=np.float)
    fp = np.sum(binary_pred[binary_target == 0] == 1, dtype=np.float)

    tn = np.sum(binary_pred[binary_target == 0] == 0, dtype=np.float)
    fn = np.sum(binary_pred[binary_target == 1] == 0, dtype=np.float)

    # from sklearn.metrics.classification import confusion_matrix
    # tn, fp, fn, tp = confusion_matrix(binary_target, binary_pred).ravel()
    return tn, fp, fn, tp


def _iou(confusion) -> float:
    tn, fp, fn, tp = confusion
    IoU = tp / np.float_(fp + tp + fn)
    return IoU


def _f1_score(confusion) -> Tuple[float, float, float]:
    tn, fp, fn, tp = confusion
    pre = tp / np.float_(tp + fp)
    rec = tp / np.float_(tp + fn)
    f1 = 2. * tp / np.float_(2. * tp + fn + fp)
    return pre, rec, f1


def _kappa(confusion) -> float:
    # confusion = (tn, fp, fn, tp)
    confusion = np.asarray(confusion, dtype=np.float).reshape(2, 2)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.float_(np.sum(sum0))

    w_mat = np.ones([n_classes, n_classes], dtype=np.int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.float_(np.sum(w_mat * expected))
    return 1 - k


def _mcc(confusion, normalized: bool) -> float:
    tn, fp, fn, tp = confusion
    mcc = (tp * tn - fp * fn) / np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )

    # normalized MCC returns a value from 0 to 1
    # if normalized: mcc = (mcc + 1.) / 2.  # solution 1
    if normalized: mcc = np.maximum(mcc, 0.)  # solution2

    return mcc


def _acc(confusion) -> float:
    tn, fp, fn, tp = confusion
    acc = (tp + tn) / np.float_(tp + fp + tn + fn)
    return acc


def _fpr(confusion) -> float:
    tn, fp, fn, tp = confusion
    fpr = fp / np.float_(fp + tn)
    return fpr


def evaluation_pixel(binary_pixel_pred: np.ndarray, binary_pixel_target: np.ndarray) -> dict:
    binary_pixel_pred = binary_pixel_pred.ravel()
    binary_pixel_target = binary_pixel_target.ravel()
    confusion = _compute_confusion_matrix(binary_pixel_target, binary_pixel_pred)

    precision, recall, dice = _f1_score(confusion)

    pixel_metrics = OrderedDict(
        Pre=precision,
        Rec=recall,
        Dice=dice,
        F1=dice,  # dice index is just the pixel F1-score
        # IoU=_iou(confusion),
        # nMCC=_mcc(confusion, normalized=True),
    )

    # pixel_metrics = OrderedDict(
    #     pixel_precision=precision,
    #     pixel_recall=recall,
    #     pixel_dice=dice,
    #     pixel_IoU=iou,
    #     pixel_mcc=mcc,
    #     pixel_kappa=_kappa(confusion),
    #     pixel_acc=_acc(confusion),
    #     pixel_fpr=_fpr(confusion),
    # )

    return pixel_metrics
