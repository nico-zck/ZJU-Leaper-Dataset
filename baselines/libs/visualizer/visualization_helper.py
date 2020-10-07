import os
from os import makedirs, path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage import transform
from sklearn import preprocessing

from .figure_tools import easy_show_image, fig2array, transparent_cmap, visualize_grid


def plot_visualization(img_name, img_src, img_pred, mask_src, mask_pred, score_pred, save_dir=None):
    fig = plt.figure(figsize=[12, 16])
    ax1 = fig.add_subplot(321)
    easy_show_image(img_src, axes=ax1)
    ax1.axis('off')
    ax1.set_title('ground-truth image')
    if img_pred is not None:
        ax2 = fig.add_subplot(322)
        easy_show_image(img_pred, axes=ax2)
        ax2.axis('off')
        ax2.set_title('predicted image')
    ax3 = fig.add_subplot(323)
    easy_show_image(mask_src, axes=ax3)
    ax3.axis('off')
    ax3.set_title('ground-truth mask')
    ax4 = fig.add_subplot(324)
    easy_show_image(mask_pred, axes=ax4)
    ax4.axis('off')
    ax4.set_title('predicted mask')
    if score_pred is not None:
        ax5 = fig.add_subplot(325)
        score_pred = easy_show_image(score_pred)
        if score_pred.ndim == 3:
            score_pred = score_pred.mean(axis=2)
        im5 = ax5.imshow(score_pred, cmap='jet')
        fig.colorbar(im5, ax=ax5)
        ax5.axis('off')
        ax5.set_title('score map, maximum: %g' % score_pred.max())
        ax6 = fig.add_subplot(326)
        _result_with_cmap(img_src, score_pred, ax6)
        ax6.set_title('mixed result')
    plt.tight_layout(pad=0)
    if save_dir:
        makedirs(save_dir, exist_ok=True)
        plt.savefig(path.join(save_dir, img_name))
    img = fig2array(fig)
    plt.close()
    return img


def _result_with_cmap(img_target, img_binary, ax=None):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=[20, 10])
        fig.subplots_adjust(top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
        ax = fig.add_subplot(111)
    easy_show_image(img_target, axes=ax)
    cmap = transparent_cmap(plt.cm.Reds)
    im = ax.imshow(img_binary, cmap=cmap)
    plt.gcf().colorbar(im, ax=ax)
    ax.axis('off')
    if ax is None:
        return fig2array(fig)


def plot_result_with_contour(img_source, img_result, index, threshold=0, save_dir=None):
    def result_with_cmap(img_source, img_result, threshold=0, ax=None):
        fig = None
        if ax is None:
            fig = plt.figure(figsize=[20, 10])
            fig.subplots_adjust(top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
            ax = fig.add_subplot(111)
        easy_show_image(img_source, axes=ax)
        cmap = transparent_cmap(plt.cm.Reds)
        if img_result.max() > 0.:
            sns.heatmap(img_result, cmap=cmap, ax=ax, cbar=True)
        else:
            sns.heatmap(img_result, cmap=cmap, center=0, ax=ax, cbar=True)
        ax.axis('off')
        if fig:
            return fig2array(fig)

    post_plot = result_with_cmap(img_source, img_result, threshold)
    from skimage import io
    white_image = np.zeros((1000, 2000, 3), np.uint8)
    white_image[:, :, :] = 255
    white_post = result_with_cmap(white_image, img_result, threshold)
    io.imsave(path.join(save_dir, 'img%s_white.png' % index), white_post)
    white_post = cv2.cvtColor(white_post, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(white_post, 254, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(post_plot, contours, -1, (255, 0, 0), 3)
    if save_dir:
        io.imsave(path.join(save_dir, 'img%s_result.png' % index), post_plot)
    else:
        return post_plot


def visualize_jet_number(mse, idx, save_dir):
    row, col = mse.shape
    fig = plt.figure(figsize=[48, 24])
    fig.subplots_adjust(top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    ax1 = fig.add_subplot(111)
    ax1.imshow(mse, cmap='jet')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    for i in range(row):
        for j in range(col):
            ax1.text(j, i, '%.3f' % mse[i][j], horizontalalignment='center', verticalalignment='center',
                     fontsize=5)
    jet_path = os.path.join(save_dir, 'img' + idx + '_jet_number.png')
    plt.savefig(jet_path)
    plt.close()


def plot_model_weights(weights, model_name, layer_name, save_dir=None):
    grid = visualize_grid(weights).squeeze()
    plt.figure(figsize=(10, 10))
    if len(grid.shape) == 3:
        plt.imshow(grid)
    elif len(grid.shape) == 2:
        plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.title('%s_%s' % (model_name, layer_name))
    if save_dir:
        plt.savefig(path.join(save_dir, '%s_%s.png' % (model_name, layer_name)))
        plt.close()
        return None
    else:
        return grid


def patches_tsne_visualization(patches_vector, patches_mask, ratio_thold=0.1, savedir=None, n_iter=5000):
    from sklearn.manifold import t_sne
    import pandas as pd
    mask_labels = []
    for mask in patches_mask:
        ratio = mask.sum() / float(mask.size)
        if ratio == 0:
            label = 'good'
        elif ratio >= ratio_thold:
            label = 'defect'
        else:
            label = 'neutral'
        mask_labels.append(label)
    mask_labels = np.array(mask_labels)
    points_num, source_dims = patches_vector.shape
    sp = sns.color_palette('muted')
    color_palette = {'good': sp[2], 'defect': sp[3], 'neutral': sp[7]}
    for perplexity in [30, 40, 50]:
        sk_tsne = t_sne.TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init='random', verbose=1)
        embedded = sk_tsne.fit_transform(patches_vector)
        df = pd.DataFrame(embedded, columns=('x', 'y'))
        df['label'] = mask_labels
        plt.figure(figsize=(12, 8))
        plt.title('Dimension:%d  Perplexity:%d  Iteration:%d' % (source_dims, perplexity, n_iter))
        ax = sns.scatterplot(x='x', y='y', hue='label', data=df, palette=color_palette)
        ax.xaxis.info.set_visible(0)
        ax.yaxis.info.set_visible(0)
        plt.savefig(path.join(savedir, 'tsne_p%d.png' % perplexity))


def vis_attn_weights(img_source, img_mask, attn_weights, save_dir, save_name):
    img_mask = transform.resize(img_mask, (64, 64))
    img_source = transform.resize(img_source, (128, 128))
    attn_weights = attn_weights.reshape((64, 64, 64, 64))
    white = np.where(img_mask != 0)
    defect_position = attn_weights[white]
    defect_position_sum = np.mean(defect_position, axis=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    defect_position_minmax = min_max_scaler.fit_transform(defect_position_sum)
    filter_map = transform.pyramid_expand(defect_position_minmax, upscale=2, sigma=10)
    fig = plt.figure(figsize=[12, 8])
    ax1 = fig.add_subplot(111)
    easy_show_image(img_source, axes=ax1)
    ax1.imshow(filter_map, alpha=0.5, cmap='gist_gray')
    ax1.axis('off')
    if not path.exists(save_dir):
        makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()
