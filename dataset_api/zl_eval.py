# -*- coding: utf-8 -*-
"""
@Time   : 2019/11/27 5:04 下午
@Author : Nico
"""
import json
import warnings
from collections import OrderedDict

import numpy as np

from .eval_pixel import evaluation_pixel
from .eval_region import evaluation_region
from .eval_sample import evaluation_sample


class ZLEval:
    """
    Evaluator for the ZJU-Leaper Dataset.
    """

    def __init__(self, binary_pixel_target: np.ndarray, binary_pixel_pred: np.ndarray):
        """
        Create an object to evaluate an inspection algorithm.
        :param binary_pixel_target: binarized pixel-wise ground truths.
        :param binary_pixel_pred: binarized pixel-wise predictions.
        """
        assert binary_pixel_pred.shape == binary_pixel_target.shape
        if np.unique(binary_pixel_pred).size == 1 or np.unique(binary_pixel_target).size == 1:
            warnings.warn('All pixel results are same, you may get meaningless metrics !')

        if binary_pixel_pred.dtype != np.bool:
            warnings.warn('Binarizing predictions!')
            binary_pixel_pred = binary_pixel_pred.astype(np.bool)
        if binary_pixel_target.dtype != np.bool:
            warnings.warn('Binarizing targets!')
            binary_pixel_target = binary_pixel_target.astype(np.bool)

        self.binary_pixel_target = binary_pixel_target
        self.binary_pixel_pred = binary_pixel_pred

    def evaluate(self) -> dict:
        """
        Calculate all metrics on the results and return metrics as a dict.
        :return:
        """
        self.metrics_dict = OrderedDict()

        pixel_metrics = evaluation_pixel(binary_pixel_pred=self.binary_pixel_pred,
                                         binary_pixel_target=self.binary_pixel_target)
        region_metrics, info_region = evaluation_region(binary_pixel_pred=self.binary_pixel_pred,
                                                        binary_pixel_target=self.binary_pixel_target,
                                                        diff_scale=True, return_info=True)
        sample_metrics = evaluation_sample(binary_pixel_pred=self.binary_pixel_pred,
                                           binary_pixel_target=self.binary_pixel_target, info_region=info_region)
        summary_score = 0.4 * pixel_metrics['Dice'] + 0.4 * region_metrics['F1'] + 0.2 * sample_metrics['F1']

        self.metrics_dict = OrderedDict(
            Pix_Dice=pixel_metrics['Dice'],
            Reg_F1=region_metrics['F1'],
            Sam_F1=sample_metrics['F1'],

            ## other metrics
            Reg_Pre_S=region_metrics['Pre_small'],
            Reg_Rec_S=region_metrics['Rec_small'],
            Reg_Pre_M=region_metrics['Pre_medium'],
            Reg_Rec_M=region_metrics['Rec_medium'],
            Reg_Pre_L=region_metrics['Pre_large'],
            Reg_Rec_L=region_metrics['Rec_large'],
            Reg_F1_S=region_metrics['F1_small'],
            Reg_F1_M=region_metrics['F1_medium'],
            Reg_F1_L=region_metrics['F1_large'],

            SCORE=summary_score,
        )

        return self.metrics_dict

    def summarize(self):
        """
        Pretty printing metrics dict
        :return:
        """
        print(json.dumps(self.metrics_dict, indent=4))
