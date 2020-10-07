import numpy as np


class ThresholdHelper():
    def __init__(self, score_pred, binary_target, metric):
        self.score_pred = score_pred
        self.binary_target = binary_target
        self.metric = metric

    def get_best_threshold(self):
        if self.metric == 'iou':
            raise NotImplementedError
        elif self.metric == 'dice':
            from sklearn.metrics import precision_recall_curve
            pres, recs, thrs = precision_recall_curve(
                y_true=self.binary_target.ravel(), probas_pred=self.score_pred.ravel(), pos_label=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                dices = 2 * (pres * recs) / (pres + recs)
            max_ind = np.nanargmax(dices)
            best_thr = thrs[max_ind]
            best_dice = dices[max_ind]
            return best_thr, best_dice
        else:
            raise NotImplementedError
