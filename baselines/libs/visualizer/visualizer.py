import os
from abc import ABC

import numpy as np
import seaborn as sns
from tqdm import tqdm

from .figure_helper import plot_IoU_figure, plot_RoC_curve, plot_hist_pixel_deviation, plot_hist_sample_score
from .visualization_helper import plot_visualization
from ..configurer import Configurer
from ..visualizer.figure_helper import plot_PR_curve


class _Visualizer(ABC):
    def __init__(self, cfg: Configurer):
        self.cfg = cfg

    def visualizing(self, model_dir: str, result_dict: dict, figure: bool, vis_rate, vis_bad: bool, **kwargs):
        sns.set()
        if figure:
            print("plotting figures...")
            fig_dir = os.path.join(model_dir, 'figuration')
            os.makedirs(fig_dir, exist_ok=True)
            self._figuration(fig_dir=fig_dir, result_dict=result_dict, **kwargs)
            print("figuration done.")
        print("visualizing results...")
        fig_dir = os.path.join(model_dir, 'visualization')
        os.makedirs(fig_dir, exist_ok=True)
        if vis_rate == 0 and vis_bad == False: return
        if vis_bad == True: vis_rate = 1
        if vis_rate != 1:
            num_result = len(next(iter(result_dict.values())))
            result_inds = list(range(0, num_result, int(num_result / (num_result * vis_rate))))
            for k, v in result_dict.items():
                if isinstance(v, np.ndarray):
                    result_dict[k] = v[result_inds]
                else:
                    result_dict[k] = [v[i] for i in result_inds]
        self._visualization(fig_dir=fig_dir, result_dict=result_dict, vis_bad=vis_bad, **kwargs)
        print('visualization done.')

    def _figuration(self, fig_dir, result_dict, **kwargs):
        flat_score = result_dict["score_pred"].ravel()
        flat_mask = result_dict["mask"].ravel()
        thresholds = plot_PR_curve(y_pred=flat_score, y_true=flat_mask, save_dir=fig_dir)
        plot_IoU_figure(flat_score=flat_score, flat_mask=flat_mask, thresholds=thresholds, save_dir=fig_dir)
        plot_RoC_curve(y_score=flat_score, y_true=flat_mask, pixel_or_sample='pixel', save_dir=fig_dir)
        plot_hist_pixel_deviation(flat_score=flat_score, flat_mask=flat_mask, save_dir=fig_dir)
        pixel_score = result_dict["score_pred"]
        num_sample = pixel_score.shape[0]
        pixel_score = pixel_score.reshape(num_sample, -1)
        inds = np.argpartition(pixel_score, kth=-16, axis=1)[:, -16:]
        sample_score = np.mean(pixel_score[np.arange(num_sample).reshape(num_sample, 1), inds], axis=1)
        sample_label = result_dict["label"]
        plot_hist_sample_score(sample_score=sample_score, sample_label=sample_label, save_dir=fig_dir)
        plot_RoC_curve(y_score=sample_score, y_true=sample_label, pixel_or_sample='sample', save_dir=fig_dir)

    def _visualization(self, fig_dir, result_dict, vis_bad, **kwargs):
        for idx in tqdm(range(len(result_dict["image"])), desc='plotting', dynamic_ncols=True):
            name = result_dict["name"][idx]
            image = result_dict["image"][idx]
            mask = result_dict["mask"][idx]
            mask_pred = result_dict["mask_pred"][idx]
            score_pred = result_dict["score_pred"][idx] if "score_pred" in result_dict else None
            image_pred = result_dict["image_pred"][idx] if "image_pred" in result_dict else None
            if vis_bad == True:
                if mask.sum() == 0:
                    if mask_pred.sum() > 10:
                        plot_dir = os.path.join(fig_dir, 'false_alarm')
                    else:
                        continue
                else:
                    if mask_pred.sum() == 0:
                        plot_dir = os.path.join(fig_dir, 'miss')
                    elif ((np.logical_and(mask, mask_pred)).sum()
                          / (np.logical_or(mask, mask_pred).sum())) < 0.3:
                        plot_dir = os.path.join(fig_dir, 'poor')
                    else:
                        continue
                plot_visualization(
                    img_name=name,
                    img_src=image, img_pred=image_pred,
                    mask_src=mask, mask_pred=mask_pred,
                    score_pred=score_pred, save_dir=plot_dir
                )
            else:
                plot_visualization(
                    img_name=name,
                    img_src=image, img_pred=image_pred,
                    mask_src=mask, mask_pred=mask_pred,
                    score_pred=score_pred, save_dir=fig_dir
                )


class ImageRecVisualizer(_Visualizer):
    pass


class ScoreMapVisualizer(_Visualizer):
    pass


class BinaryMapVisualizer(_Visualizer):
    def _figuration(self, fig_dir, result_dict, **kwargs):
        flat_mask = result_dict["mask"].ravel()
        flat_pred = result_dict["mask_pred"].ravel()
        pass
        pixel_label = result_dict["mask_pred"]
        num_sample = pixel_label.shape[0]
        pixel_label = pixel_label.reshape(num_sample, -1)
        sample_score = np.sum(pixel_label, axis=1)
        sample_label = result_dict["label"]
        plot_hist_sample_score(sample_score=sample_score, sample_label=sample_label, save_dir=fig_dir)
        plot_RoC_curve(y_score=sample_score, y_true=sample_label, pixel_or_sample='sample', save_dir=fig_dir)
