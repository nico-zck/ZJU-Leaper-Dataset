import math
from os import path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.autograd import Variable

from .figure_tools import fig2array


def _compute_confusion_matrix(binary_target, binary_pred):
    tp = np.sum(binary_pred[binary_target == 1] == 1, dtype=np.float)
    fp = np.sum(binary_pred[binary_target == 0] == 1, dtype=np.float)
    tn = np.sum(binary_pred[binary_target == 0] == 0, dtype=np.float)
    fn = np.sum(binary_pred[binary_target == 1] == 0, dtype=np.float)
    return tn, fp, fn, tp


def _iou(confusion):
    tn, fp, fn, tp = confusion
    IoU = tp / (fp + tp + fn)
    return IoU


def plot_hist_pixel_deviation(flat_score, flat_mask, save_dir, bins=None):
    assert flat_score.size == flat_mask.size
    score_normal = flat_score[~flat_mask]
    score_defective = flat_score[flat_mask]
    score_normal = np.log10(score_normal.ravel() + 1e-16)
    score_defective = np.log10(score_defective.ravel() + 1e-16)
    mean_normal, mean_defective = score_normal.mean(), score_defective.mean()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    sns.distplot(score_normal, bins=bins, kde=True, norm_hist=True, color='mediumseagreen', label='Good',
                 hist_kws=dict(histtype='bar', cumulative=False), ax=ax1)
    sns.distplot(score_defective, bins=bins, kde=True, norm_hist=True, color='salmon', label='Defect',
                 hist_kws=dict(histtype='bar', cumulative=False), ax=ax1)
    ax1.legend(loc='upper right')
    ax1.set_title('Log-Distribution of pixel score')
    ax1.set_xlabel('Log-score')
    ax1.set_ylabel('Density')
    ax2 = fig.add_subplot(212)
    sns.distplot(score_normal, bins=bins, kde=False, norm_hist=True,
                 hist_kws=dict(histtype='stepfilled', cumulative=-1), color='mediumseagreen',
                 label='Good', ax=ax2)
    sns.distplot(score_defective, bins=bins, kde=False, norm_hist=True,
                 hist_kws=dict(histtype='stepfilled', cumulative=1), color='salmon',
                 label='Defect', ax=ax2)
    ax2.legend(loc='upper right')
    ymin, ymax = ax2.get_ylim()
    ax2.vlines(mean_normal, ymin, ymax, colors='g', linestyles='dashed')
    ax2.vlines(mean_defective, ymin, ymax, colors='r', linestyles='dashed')
    ax2.annotate('', xy=(mean_normal, ymax * 0.3), xycoords='data',
                 xytext=(mean_defective, ymax * 0.3), textcoords='data',
                 arrowprops={'arrowstyle': '<->'})
    ax2.annotate('%.4f' % (mean_defective - mean_normal),
                 xy=((mean_normal + mean_defective) / 2, ymax * 0.3), xycoords='data', rotation=-90,
                 xytext=(0, 2), textcoords='offset points')
    ax2.set_title('Cumulative distribution of pixel score')
    ax2.set_xlabel('Log-score')
    ax2.set_ylabel('Probability')
    plt.tight_layout()
    plt.savefig(path.join(save_dir, 'hist_pixel.png'))
    plt.close()


def plot_hist_sample_score(sample_score, sample_label, save_dir):
    good_score = sample_score[sample_label == 0]
    bad_score = sample_score[sample_label == 1]
    ax2 = plt.gca()
    sns.distplot(good_score, bins=np.unique(good_score).size, kde=False, norm_hist=True,
                 hist_kws=dict(histtype='stepfilled', cumulative=-1),
                 color='mediumseagreen', label='Good', ax=ax2)
    sns.distplot(bad_score, bins=np.unique(bad_score).size, kde=False, norm_hist=True,
                 hist_kws=dict(histtype='stepfilled', cumulative=1),
                 color='salmon', label='Bad', ax=ax2)
    ax2.legend(loc='lower right')
    ax2.set_title('Cumulative distribution of sample score')
    ax2.set_xlabel('Sample score')
    ax2.set_ylabel('Probability')
    plt.savefig(path.join(save_dir, 'hist_sample.png'))
    plt.close()


def plot_samples_hist_layer_wise(samples_record, model_name, save_dir=None):
    neg = samples_record['neg']
    pos = samples_record['pos']
    pos_max_error_l2 = np.array(pos['max_error_l2'])
    neg_max_error_l2 = np.array(neg['max_error_l2'])
    pos_max_post = np.array(pos['max_post'])
    neg_max_post = np.array(neg['max_post'])
    if len(pos_max_error_l2.shape) == len(neg_max_post.shape) == 1:
        pos_max_error_l2 = pos_max_error_l2[..., None]
        neg_max_error_l2 = neg_max_error_l2[..., None]
        pos_max_post = pos_max_post[..., None]
        neg_max_post = neg_max_post[..., None]
    samples_num = pos_max_error_l2.shape[0]
    layers_num = pos_max_error_l2.shape[1]
    for l_index in range(layers_num):
        fig = plt.figure(figsize=[12, 5])
        ax1 = fig.add_subplot(121)
        sns.distplot(pos_max_error_l2[:, l_index], bins=samples_num, kde=False,
                     hist_kws=dict(histtype='stepfilled', cumulative=-1), color='g',
                     label='Pos', ax=ax1)
        sns.distplot(neg_max_error_l2[:, l_index], bins=samples_num, kde=False,
                     hist_kws=dict(histtype='stepfilled', cumulative=1), color='r',
                     label='Neg', ax=ax1)
        ax1.legend(loc='lower right')
        ax1.set_title('Max MSE')
        ax1.set_xlabel('MSE')
        ax1.set_ylabel('Count')
        ax2 = fig.add_subplot(122)
        sns.distplot(pos_max_post[:, l_index], bins=samples_num, kde=False,
                     hist_kws=dict(histtype='stepfilled', cumulative=-1), color='g', label='Pos',
                     ax=ax2)
        sns.distplot(neg_max_post[:, l_index], bins=samples_num, kde=False,
                     hist_kws=dict(histtype='stepfilled', cumulative=1), color='r', label='Neg',
                     ax=ax2)
        ax2.legend(loc='lower right')
        ax2.set_title('Max Score')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Count')
        plt.suptitle(model_name)
        if save_dir:
            plt.savefig(path.join(save_dir, '%s_%d_hist_sample.png' % (model_name, l_index)))
            img_array = None
        else:
            img_array = fig2array(fig)
        plt.close()
        return img_array


def plot_PR_curve(y_pred, y_true, save_dir):
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_pred, pos_label=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    max_idx = np.nanargmax(f1_scores)
    line = recalls[::-1]
    ind = np.argwhere(np.diff(np.sign(line - precisions[::-1]))).ravel()[-1]
    bep = line[ind]
    auc = metrics.auc(recalls, precisions)
    plt.step(recalls, precisions, where='post')
    plt.fill_between(recalls, precisions, alpha=0.3, step='post')
    plt.plot(line, line, '--', alpha=0.5, )
    plt.plot(bep, bep, 'ro')
    plt.annotate(f'{bep.item():.3f}', (bep, bep))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.suptitle('Best F1: %.3f, Pre: %.3f, Rec: %.3f Threshold: %.3f \n BEP: %.3f, AUC: %.3f'
                 % (f1_scores[max_idx], precisions[max_idx], recalls[max_idx], thresholds[max_idx], bep, auc))
    plt.savefig(path.join(save_dir, 'PR_curve.png'))
    plt.close()
    return thresholds


def plot_IoU_figure(flat_score, flat_mask, thresholds, save_dir, max_num_ths=200):
    num_ths = len(thresholds)
    if num_ths > max_num_ths:
        th_idx = np.linspace(0, num_ths - 1, max_num_ths, dtype=np.int)
        thresholds = thresholds[th_idx]
    from concurrent.futures import ThreadPoolExecutor
    def foo(pred, mask, th):
        pred_binary = np.asarray(pred >= th)
        confusion = _compute_confusion_matrix(binary_target=mask, binary_pred=pred_binary)
        IoU = _iou(confusion)
        return IoU

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(foo, flat_score, flat_mask, th) for th in thresholds]
    IoU_list = [f.result() for f in futures]
    IoU_list = np.asarray(IoU_list)
    max_idx = np.nanargmax(IoU_list)
    max_IoU = IoU_list[max_idx]
    max_IoU_th = thresholds[max_idx]
    fig = plt.figure()
    plt.plot(thresholds, IoU_list)
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.suptitle('Max-IoU: %.4f, Threshold: %.4f' % (max_IoU, max_IoU_th))
    plt.savefig(path.join(save_dir, 'IoU.png'))
    plt.close()


def plot_RoC_curve(y_score, y_true, pixel_or_sample, save_dir):
    assert pixel_or_sample in ['pixel', 'sample']
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--', alpha=0.5)
    plt.xlabel('False alarm')
    plt.ylabel('Recall')
    plt.suptitle(f'{pixel_or_sample} RoC, AUC: {auc:.4f}')
    plt.savefig(path.join(save_dir, 'RoC_curve_%s.png' % (pixel_or_sample)))
    plt.close()


def draw_multiple_ROC(fpr_list, tpr_list, line_name_list, save_path):
    line_style = ['r-', 'b-', 'g-', 'k-', 'm-', 'c-', 'y-', 'r--', 'b--', 'g--', 'k--']
    style_number = len(line_style)
    roc_order = 0
    plt.figure(figsize=(5, 4))
    for i in range(len(line_name_list)):
        cur_fpr = fpr_list[i]
        cur_tpr = tpr_list[i]
        plt.plot(cur_fpr, cur_tpr, line_style[(roc_order % style_number)], linewidth=2)
        roc_order += 1
    plt.title('roc curve')
    plt.xlabel('false alarm')
    plt.ylabel('recall')
    plt.legend(line_name_list, loc=4)
    plt.savefig(save_path)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding=window_size / 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size / 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size / 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size / 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size / 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
