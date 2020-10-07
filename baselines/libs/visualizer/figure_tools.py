import numpy as np
from PIL import Image


def easy_show_image(image, axes=None):
    assert image.ndim in [2, 3]
    if image.min() < 0 or (1 < image.max() < 5):
        '''matplotlib only receive value in the range [0, 1] or [0, 255]'''
        import warnings
        warnings.warn('a image will be scaled into [0,1], min and max are: %g, %g' % (image.min(), image.max()))
        image = (image - image.min()) / (image.max() - image.min())
    image = image.squeeze()
    if image.ndim == 3:
        c_idx = np.argmin(image.shape)
        c = image.shape[c_idx]
        if c != 3:
            raise ValueError('the channel of image is wrong !')
        if c_idx == 0:
            image = np.transpose(image, [1, 2, 0])
        if axes:
            axes.imshow(image)
        else:
            return image
    elif image.ndim == 2:
        if axes:
            axes.imshow(image, cmap='gray')
        else:
            return image
    else:
        raise NotImplementedError


def transparent_cmap(cmap):
    my_cmap = cmap
    my_cmap._init()
    color_bar = np.linspace(0.1, 0.7, 259)
    color_bar[0] = 0.0
    my_cmap._lut[:, -1] = color_bar
    return my_cmap


def fig2array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    buf = fig2array(fig)
    w, h, d = buf.shape
    return Image.fromstring("RGBA", (w, h), buf.tostring())


def visualize_grid(weights, ubound=255.0, padding=2):
    N, H, W, C = weights.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = weights[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def crop_mask_size(foo, gt_mask):
    assert len(foo.shape) == len(gt_mask.shape)
    assert foo.shape[0] == gt_mask.shape[0] and all(foo.shape <= gt_mask.shape)
    target_shape = foo.shape
    gt_mask = gt_mask[:, :target_shape[1], :target_shape[2]]
    return gt_mask


def flatten_results_and_masks(results_list, masks_list):
    flat_result = np.concatenate([r.flat for r in results_list])
    flat_mask = np.concatenate([m.flat for m in masks_list])
    assert flat_result.shape == flat_mask.shape
    return flat_result, flat_mask


def cut_edge(img, edge_size):
    cutted_img = img[edge_size:-edge_size, edge_size:-edge_size]
    return cutted_img
