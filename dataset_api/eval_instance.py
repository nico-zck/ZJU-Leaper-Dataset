# -*- coding: utf-8 -*-
"""
@Time   : 2019/11/17 4:41 下午
@Author : Nico
"""
import itertools
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties

# define the threshold of overlap ratio to determine valid region proposals.
OVERLAP_THRESH = 0.51

# define the thresholds for averaging instance precision and recall
IOU_MIN = 0.1
IOU_MAX = 0.5
IOU_STEP = 0.05

# define the size of small, medium, large defects
DEFAULT_IMG_SIZE = 512
DEFAULT_SMALL_TH = 50 ** 2
DEFAULT_MEDIUM_TH = 100 ** 2
SMALL_TH = MEDIUM_TH = 0  # will be changed for different image size


def _generate_bbox_given_map(binary_map: np.ndarray) -> np.ndarray:
    """
    Generate corresponding bboxes given binary maps.
    :param binary_map:
    :return:
    """
    num_img = len(binary_map)
    bboxes = []
    bbox_id = 0
    for img_id in range(num_img):
        binary_img = binary_map[img_id]
        img_label = label(binary_img)
        skbboxes = regionprops(img_label, cache=True)  # sk means skimage
        for skbbox in skbboxes:
            bbox = [(img_id, bbox_id, skbbox)]
            bboxes.append(bbox)
            bbox_id += 1
    '''columns: image_id, bbox_id, skbbox_object'''
    bboxes = np.array(list(itertools.chain(*bboxes)))
    return bboxes


def _generate_region_map_given_map_and_bbox(img_binary_map, sk_bbox: RegionProperties):
    rgn_binary_map = np.zeros_like(img_binary_map, dtype=np.bool)
    y1, x1, y2, x2 = sk_bbox.bbox
    rgn_binary_map[y1:y2, x1:x2] = sk_bbox.image
    return rgn_binary_map


def _compute_bbox_overlap(cord_pred, cord_target):
    """
    The ratio of the overlapped rectangle area to the predicted bbox area
    :param cord_pred:
    :param cord_target:
    :return:
    """
    ay1, ax1, ay2, ax2 = cord_pred
    by1, bx1, by2, bx2 = cord_target
    if (ay1 <= by2 and ay2 >= by1) and (ax1 <= bx2 and ax2 >= bx1):
        height = min(ay2, by2) - max(ay1, by1)
        width = min(ax2, bx2) - max(ax1, bx1)
        area_overlap = height * width
    else:
        area_overlap = 0
    return area_overlap


def _region_info_pred(bboxes_pred: np.ndarray) -> pd.DataFrame:
    info_pred = []
    for bbox_pred in bboxes_pred:
        id_img, id_pred, skbbox_pred = bbox_pred
        area_pred = skbbox_pred.image.sum()
        info_pred.append(dict(id_pred=id_pred, area_pred=area_pred, id_img=id_img))
    info_pred = pd.DataFrame(info_pred)
    return info_pred


def _region_info_target(bboxes_target: np.ndarray) -> pd.DataFrame:
    info_target = []
    for bbox_target in bboxes_target:
        id_img, id_target, skbbox_target = bbox_target
        area_target = skbbox_target.image.sum()
        info_target.append(dict(id_target=id_target, area_target=area_target, id_img=id_img))
    info_target = pd.DataFrame(info_target)
    return info_target


def _region_info_overlap(binary_pred: np.ndarray, binary_target: np.ndarray,
                         bboxes_pred: np.ndarray, bboxes_target: np.ndarray) -> pd.DataFrame:
    """
    Maintain a table that records the relation between predicted regions (region proposals) and target regions.
    :param binary_pred:
    :param binary_target:
    :param pred_bboxes:
    :param target_bboxes:
    :return:
    """
    num_imgs = len(binary_pred)

    info_overlap = []
    for id_img in range(num_imgs):
        img_binary_pred = binary_pred[id_img]
        img_binary_target = binary_target[id_img]
        img_bboxes_pred = bboxes_pred[bboxes_pred[:, 0] == id_img]
        img_bboxes_target = bboxes_target[bboxes_target[:, 0] == id_img]

        num_pred = len(img_bboxes_pred)
        num_target = len(img_bboxes_target)

        if num_pred == 0 or num_target == 0:
            continue
        for i in range(num_pred):
            for j in range(num_target):
                _, id_pred, skbbox_pred = img_bboxes_pred[i]
                _, id_target, skbbox_target = img_bboxes_target[j]

                # Check bbox overlap first, because if two regions overlap each other,
                #   their bounding-boxes must be overlapped, but not vice versa.
                bbox_overlap = _compute_bbox_overlap(cord_pred=skbbox_pred.bbox, cord_target=skbbox_target.bbox)
                if bbox_overlap == 0: continue  # not overlap at bbox level

                rgn_binary_pred = _generate_region_map_given_map_and_bbox(
                    img_binary_map=img_binary_pred, sk_bbox=skbbox_pred)
                rgn_binary_target = _generate_region_map_given_map_and_bbox(
                    img_binary_map=img_binary_target, sk_bbox=skbbox_target)
                assert rgn_binary_pred.shape == rgn_binary_target.shape
                assert rgn_binary_pred.dtype == rgn_binary_target.dtype == np.bool
                area_overlap = np.sum(np.logical_and(rgn_binary_pred, rgn_binary_target))
                if area_overlap == 0: continue  # not overlap at region level

                area_pred = skbbox_pred.image.sum()
                area_target = skbbox_target.image.sum()

                area_union = np.sum(np.logical_or(rgn_binary_pred, rgn_binary_target))
                ratio_overlap = area_overlap / float(area_pred)
                iou = area_overlap / float(area_union)

                info_overlap.append(dict(
                    id_pred=id_pred, id_target=id_target, area_pred=area_pred, area_target=area_target,
                    area_union=area_union, area_overlap=area_overlap, ratio_overlap=ratio_overlap, iou=iou,
                    id_img=id_img
                ))
    if len(info_overlap) == 0:
        # create a empty DF to avoid merging error
        info_overlap = pd.DataFrame(columns=['id_pred', 'id_target', 'area_pred', 'area_target', 'area_union',
                                             'area_overlap', 'ratio_overlap', 'iou', 'id_img'])
    else:
        info_overlap = pd.DataFrame(info_overlap)
    return info_overlap


def _precision_for_region_pred(info_pred_overlap: pd.DataFrame, iou_thresh: float, scale: str = 'all') -> float:
    """
    compute the precision for predicted regions
    :param info_pred_overlap:
    :param iou_thresh:
    :return:
    """
    if scale == 'all':
        pass
    elif scale == 'small':
        info_pred_overlap = info_pred_overlap[
            (info_pred_overlap['area_target'] <= SMALL_TH)
            | (info_pred_overlap['id_target'].isna() & (info_pred_overlap['area_pred'] <= SMALL_TH))]
    elif scale == 'medium':
        info_pred_overlap = info_pred_overlap[
            ((SMALL_TH < info_pred_overlap['area_target']) & (info_pred_overlap['area_target'] < MEDIUM_TH))
            | (info_pred_overlap['id_target'].isna()
               & ((SMALL_TH < info_pred_overlap['area_pred']) & (info_pred_overlap['area_pred'] < MEDIUM_TH)))]
    elif scale == 'large':
        info_pred_overlap = info_pred_overlap[
            (info_pred_overlap['area_target'] >= MEDIUM_TH)
            | (info_pred_overlap['id_target'].isna() & (info_pred_overlap['area_pred'] >= MEDIUM_TH))]
    else:
        raise NotImplementedError

    info_valid = info_pred_overlap[
        (info_pred_overlap['iou'] > iou_thresh) | (info_pred_overlap['ratio_overlap'] > OVERLAP_THRESH)]
    num_valid = len(info_valid['id_pred'].unique())
    num_pred = len(info_pred_overlap['id_pred'].unique())
    try:
        precision = num_valid / float(num_pred)
    except ZeroDivisionError:
        precision = np.nan
    return precision


def _recall_for_region_target(info_target_overlap: pd.DataFrame, iou_thresh: float, scale: str = 'all') -> float:
    """
    compute the recall for target regions
    :param info_target_overlap:
    :param iou_thresh:
    :return:
    """
    if scale == 'all':
        pass
    elif scale == 'small':
        info_target_overlap = \
            info_target_overlap[info_target_overlap['area_target'] <= SMALL_TH]
    elif scale == 'medium':
        info_target_overlap = \
            info_target_overlap[(SMALL_TH < info_target_overlap['area_target'])
                                & (info_target_overlap['area_target'] < MEDIUM_TH)]
    elif scale == 'large':
        info_target_overlap = \
            info_target_overlap[info_target_overlap['area_target'] >= MEDIUM_TH]
    else:
        raise NotImplementedError

    # this time need to calculate overlap based on the ground-truth regions.
    gb = info_target_overlap.groupby('id_target', as_index=False)
    info_target_ = gb.agg({'area_target': 'first', 'area_pred': 'sum', 'area_overlap': 'sum', 'id_img': 'first'})
    info_target_['area_union'] = info_target_['area_target'] + info_target_['area_pred'] - info_target_['area_overlap']
    info_target_['ratio_overlap'] = info_target_['area_overlap'] / info_target_['area_target']
    info_target_['iou'] = info_target_['area_overlap'] / info_target_['area_union']

    info_valid = info_target_[(info_target_['iou'] > iou_thresh) | (info_target_['ratio_overlap'] > OVERLAP_THRESH)]
    num_valid = len(info_valid['id_target'].unique())
    num_target = len(info_target_['id_target'])
    try:
        recall = num_valid / float(num_target)
    except ZeroDivisionError:
        recall = np.nan
    return recall


def evaluation_instance(binary_pred: np.ndarray, binary_target: np.ndarray, diff_scale: bool = True):
    binary_pred = binary_pred.squeeze()
    binary_target = binary_target.squeeze()
    assert binary_pred.shape == binary_target.shape

    _, h, w = binary_target.shape
    global SMALL_TH, MEDIUM_TH
    SMALL_TH = DEFAULT_SMALL_TH * (float(h) / DEFAULT_IMG_SIZE) * (float(w) / DEFAULT_IMG_SIZE)
    MEDIUM_TH = DEFAULT_MEDIUM_TH * (float(h) / DEFAULT_IMG_SIZE) * (float(w) / DEFAULT_IMG_SIZE)

    if np.unique(binary_pred).size == 1:
        warnings.warn('All pixel results are same, you may get meaningless metrics !')

    if binary_pred.dtype != np.bool:
        binary_pred = binary_pred.astype(np.bool)
    if binary_target.dtype != np.bool:
        binary_target = binary_target.astype(np.bool)

    ##### generate rectangles based on the regions
    '''columns: img_id, box_id, skbbox_object'''
    bboxes_pred = _generate_bbox_given_map(binary_pred)
    bboxes_target = _generate_bbox_given_map(binary_target)
    # assert there is at least one defective prediction
    if bboxes_pred.size == 0 or bboxes_target.size == 0:
        warnings.warn("Instance evaluation got meaningless result !")
        stats_dict = OrderedDict({
            'iAP': np.nan,
            'iAR': np.nan,
        })
        if diff_scale:
            stats_dict.update({
                'iAP_SMALL': np.nan,
                'iAR_SMALL': np.nan,
                'iAP_MEDIUM': np.nan,
                'iAR_MEDIUM': np.nan,
                'iAP_LARGE': np.nan,
                'iAR_LARGE': np.nan,
            })
        return stats_dict

    ##### compute information of predicted regions (region proposals) and target regions
    info_pred = _region_info_pred(bboxes_pred=bboxes_pred)
    info_target = _region_info_target(bboxes_target=bboxes_target)
    info_overlap = _region_info_overlap(binary_pred=binary_pred, binary_target=binary_target,
                                        bboxes_pred=bboxes_pred, bboxes_target=bboxes_target)

    info_pred_overlap = pd.merge(info_pred, info_overlap, 'left')
    info_target_overlap = pd.merge(info_target, info_overlap, 'left')

    ##### compute the average precision and recall based on the PR curve
    # defect detection task not requires high IoUs
    iou_thrs = np.linspace(IOU_MIN, IOU_MAX, round((IOU_MAX - IOU_MIN) / IOU_STEP) + 1, endpoint=True)

    # initialize the dict of evaluator metrics
    pre_rec_list = []

    ##### compute the instance precision and recall on the certain IoU threshold
    for iou_th in iou_thrs:
        pre_rec_ins = OrderedDict({'IoU': iou_th})

        pre_rec_ins['PRE'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=iou_th)
        pre_rec_ins['REC'] = _recall_for_region_target(info_target_overlap, iou_thresh=iou_th)

        if diff_scale:
            pre_rec_ins['PRE_SMALL'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=iou_th, scale='small')
            pre_rec_ins['REC_SMALL'] = _recall_for_region_target(info_target_overlap, iou_thresh=iou_th, scale='small')

            pre_rec_ins['PRE_MEDIUM'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=iou_th, scale='medium')
            pre_rec_ins['REC_MEDIUM'] = _recall_for_region_target(info_target_overlap, iou_thresh=iou_th,
                                                                  scale='medium')

            pre_rec_ins['PRE_LARGE'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=iou_th, scale='large')
            pre_rec_ins['REC_LARGE'] = _recall_for_region_target(info_target_overlap, iou_thresh=iou_th, scale='large')

        pre_rec_list.append(pre_rec_ins)
    # put these result in the dataframe and compute the mean of these metrics
    metrics_df = pd.DataFrame(pre_rec_list, columns=list(pre_rec_list[0].keys()))
    avg_pre_rec_ins = metrics_df.mean()

    # save final metrics
    stats_dict = OrderedDict({
        'iAP': avg_pre_rec_ins['PRE'],  # instance Average Precision
        'iAR': avg_pre_rec_ins['REC'],  # instance Average Recall
    })
    if diff_scale:
        stats_dict.update({
            'iAP_SMALL': avg_pre_rec_ins['PRE_SMALL'],
            'iAR_SMALL': avg_pre_rec_ins['REC_SMALL'],
            'iAP_MEDIUM': avg_pre_rec_ins['PRE_MEDIUM'],
            'iAR_MEDIUM': avg_pre_rec_ins['REC_MEDIUM'],
            'iAP_LARGE': avg_pre_rec_ins['PRE_LARGE'],
            'iAR_LARGE': avg_pre_rec_ins['REC_LARGE'],
        })
    return stats_dict
