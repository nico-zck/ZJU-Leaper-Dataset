# -*- coding: utf-8 -*-
"""
@Time   : 2019/11/17 4:41 下午
@Author : Nico
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties

# Define the threshold of overlap ratio to determine valid region proposals.
OVERLAP_THRESH = 0.51

# Define the thresholds for averaging instance precision and recall. Of note, defect detection task not requires high IoUs
# IOUs = np.r_[0.1:0.5:0.05]
IOUs = 0.3

# Define the size of small, medium, large defects
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
    id_bbox = 0
    # id_bbox and id_img start from 0
    for id_img in range(num_img):
        binary_img = binary_map[id_img]
        img_label = label(binary_img)
        skbboxes = regionprops(img_label, cache=True)  # sk means skimage
        # skbboxes will be [] if the image is empty
        for skbbox in skbboxes:
            bbox = [(id_img, id_bbox, skbbox)]
            bboxes.append(bbox)
            id_bbox += 1
    '''columns: id_img, id_bbox, skbbox_object'''
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
    if bboxes_pred.size > 0:
        for bbox_pred in bboxes_pred:
            id_img, id_pred, skbbox_pred = bbox_pred
            area_pred = skbbox_pred.image.sum()
            info_pred.append(dict(id_pred=id_pred, area_pred=area_pred, id_img=id_img))
    info_pred = pd.DataFrame(info_pred)
    return info_pred


def _region_info_target(bboxes_target: np.ndarray) -> pd.DataFrame:
    info_target = []
    if bboxes_target.size > 0:
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
                ratio_overlap = area_overlap / np.float_(area_pred)
                iou = area_overlap / np.float_(area_union)

                info_overlap.append(dict(
                    id_pred=id_pred, id_target=id_target, area_pred=area_pred, area_target=area_target,
                    area_union=area_union, area_overlap=area_overlap, ratio_overlap=ratio_overlap, iou=iou,
                    id_img=id_img
                ))
    if len(info_overlap) == 0:
        # create a empty DF to avoid merging error
        info_overlap = pd.DataFrame(columns=['id_pred', 'id_target', 'id_img',
                                             'area_pred', 'area_target', 'area_union',
                                             'area_overlap', 'ratio_overlap', 'iou'])
    else:
        info_overlap = pd.DataFrame(info_overlap)
    return info_overlap


def _precision_for_region_pred(info_pred_overlap: pd.DataFrame, iou_thresh: float, ol_thresh: float,
                               scale: str = 'all') -> float:
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

    info_pred_valid = info_pred_overlap[
        (info_pred_overlap['iou'] > iou_thresh) | (info_pred_overlap['ratio_overlap'] > ol_thresh)]
    num_valid = len(info_pred_valid['id_pred'].unique())
    num_pred = len(info_pred_overlap['id_pred'].unique())
    precision = num_valid / np.float_(num_pred)
    return precision


def _recall_for_region_target(info_target_overlap: pd.DataFrame, iou_thresh: float, ol_thresh: float,
                              scale: str = 'all') -> float:
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

    info_target_valid = info_target_[
        (info_target_['iou'] > iou_thresh) | (info_target_['ratio_overlap'] > ol_thresh)]
    num_valid = len(info_target_valid['id_target'].unique())
    num_target = len(info_target_['id_target'])

    recall = num_valid / np.float_(num_target)
    return recall


def __f1(pre, rec):
    return 2. * (pre * rec) / np.float_(pre + rec)


def evaluation_region(binary_pixel_pred: np.ndarray, binary_pixel_target: np.ndarray,
                      diff_scale: bool = False, return_info: bool = False):
    binary_pixel_pred = binary_pixel_pred.squeeze()
    binary_pixel_target = binary_pixel_target.squeeze()

    _, h, w = binary_pixel_target.shape
    global SMALL_TH, MEDIUM_TH
    SMALL_TH = DEFAULT_SMALL_TH * (np.float_(h) / DEFAULT_IMG_SIZE) * (np.float_(w) / DEFAULT_IMG_SIZE)
    MEDIUM_TH = DEFAULT_MEDIUM_TH * (np.float_(h) / DEFAULT_IMG_SIZE) * (np.float_(w) / DEFAULT_IMG_SIZE)

    ##### generate rectangles based on the regions
    # columns: img_id, box_id, skbbox_object
    bboxes_pred = _generate_bbox_given_map(binary_pixel_pred)
    bboxes_target = _generate_bbox_given_map(binary_pixel_target)
    # compute information of predicted regions (region proposals) and target regions
    # columns: id_img, id_pred/id_target, bbox
    info_pred = _region_info_pred(bboxes_pred=bboxes_pred)
    info_target = _region_info_target(bboxes_target=bboxes_target)

    ##### calculate overlap of regions
    ## this requires at least one region for all predictions or all targets
    if info_pred.empty or info_target.empty:
        warnings.warn("Instance evaluation got meaningless result !")
        region_metrics = dict.fromkeys([
            'Pre', 'Rec', 'F1',
            'Pre_small', 'Rec_small', 'F1_small',
            'Pre_medium', 'Rec_medium', 'F1_medium',
            'Pre_large', 'Rec_large', 'F1_large'
        ], np.nan)
        if return_info:
            region_info = dict(info_pred=info_pred, info_target=info_target, info_overlap=None)
            return region_metrics, region_info
        else:
            return region_metrics
    ## then calculate overlap of regions
    # columns=['id_pred', 'id_target', 'id_img', 'area_pred', 'area_target', 'area_union', 'area_overlap', 'ratio_overlap', 'iou']
    info_overlap = _region_info_overlap(binary_pred=binary_pixel_pred, binary_target=binary_pixel_target,
                                        bboxes_pred=bboxes_pred, bboxes_target=bboxes_target)

    info_pred_overlap = pd.merge(info_pred, info_overlap, 'left')
    info_target_overlap = pd.merge(info_target, info_overlap, 'left')

    ##### compute the instance precision and recall on the certain IoU threshold
    region_metrics = {}
    region_metrics['Pre'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=IOUs,
                                                       ol_thresh=OVERLAP_THRESH)
    region_metrics['Rec'] = _recall_for_region_target(info_target_overlap, iou_thresh=IOUs,
                                                      ol_thresh=OVERLAP_THRESH)
    region_metrics['F1'] = __f1(region_metrics['Pre'], region_metrics['Rec'])
    if diff_scale:
        region_metrics['Pre_small'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=IOUs,
                                                                 ol_thresh=OVERLAP_THRESH, scale='small')
        region_metrics['Rec_small'] = _recall_for_region_target(info_target_overlap, iou_thresh=IOUs,
                                                                ol_thresh=OVERLAP_THRESH, scale='small')
        region_metrics['Pre_medium'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=IOUs,
                                                                  ol_thresh=OVERLAP_THRESH, scale='medium')
        region_metrics['Rec_medium'] = _recall_for_region_target(info_target_overlap, iou_thresh=IOUs,
                                                                 ol_thresh=OVERLAP_THRESH, scale='medium')
        region_metrics['Pre_large'] = _precision_for_region_pred(info_pred_overlap, iou_thresh=IOUs,
                                                                 ol_thresh=OVERLAP_THRESH, scale='large')
        region_metrics['Rec_large'] = _recall_for_region_target(info_target_overlap, iou_thresh=IOUs,
                                                                ol_thresh=OVERLAP_THRESH, scale='large')
        region_metrics['F1_small'] = __f1(region_metrics['Pre_small'], region_metrics['Rec_small'])
        region_metrics['F1_medium'] = __f1(region_metrics['Pre_medium'], region_metrics['Rec_medium'])
        region_metrics['F1_large'] = __f1(region_metrics['Pre_large'], region_metrics['Rec_large'])

    ##### return metrics
    if return_info:
        region_info = dict(info_pred=info_pred, info_target=info_target, info_overlap=info_overlap)
        return region_metrics, region_info
    else:
        return region_metrics
