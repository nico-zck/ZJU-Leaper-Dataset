import numpy as np
from skimage.transform import resize

try:
    from sklearn.feature_extraction.image import _extract_patches
except ImportError:
    from sklearn.feature_extraction.image import extract_patches as _extract_patches


def noise_injection(x, noise_kwargs):
    from skimage.util.noise import random_noise
    channel = x.shape[2]
    if channel == 1 or channel == 3:
        y = random_noise(x, **noise_kwargs)
    else:
        raise RuntimeError('Channel is not grayscale or RGB')
    return y


def histogram_equalization_image(img):
    from skimage import exposure
    for c in range(img.shape[-1]):
        img[:, :, c] = exposure.equalize_hist(img[:, :, c])
    return img


def illumination_normalization_image(img, alpha=4.):
    if alpha is None:
        alpha = 4.0
    kernel1 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    kernel2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    from scipy.ndimage.filters import convolve
    img1 = convolve(img, kernel1)
    img2 = convolve(img, kernel2)
    img = np.arctan(alpha * (img1 / img2))
    img = (img - img.min()) / (img.max() - img.min())
    return img


def zca_whitening_patch(X, epsilon=1e-5):
    shape = X.shape
    X = X.reshape(X.shape[0], -1)
    sigma = np.cov(X, rowvar=False)
    U, S, V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    X = np.dot(X, ZCAMatrix)
    X = X.reshape(*shape)
    return X


def zca_whitening(X, epsilon=1e-5):
    sigma = np.cov(X, rowvar=False)
    U, S, V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    return np.dot(X, ZCAMatrix)


def block_mean(preliminary_result, block_size, block_stride=None):
    preliminary_result = preliminary_result.squeeze()
    if block_stride:
        assert block_stride <= block_size
    else:
        block_stride = block_size
    if block_size > 1:
        patch_post = _extract_patches(preliminary_result, patch_shape=block_size, extraction_step=block_stride)
        patch_post = patch_post.mean(axis=(2, 3))
        img_post = resize(patch_post, preliminary_result.shape, order=0, mode='constant', anti_aliasing=False)
    else:
        img_post = preliminary_result
    return img_post


def block_TV(preliminary_result, block_size, block_stride=None):
    assert block_size > 1
    if block_stride:
        assert block_stride <= block_size
    else:
        block_stride = block_size
    patch_post = _extract_patches(preliminary_result, patch_shape=block_size, extraction_step=block_stride)

    def patch_TV(patches):
        row, col, height, width = patches.shape
        tv = np.zeros([row, col])
        for r in range(row):
            for c in range(col):
                tmp = np.sqrt(
                    (patches[r, c, 1:, :-1] - patches[r, c, :-1, :-1]) ** 2
                    + (patches[r, c, :-1, 1:] - patches[r, c, :-1, :-1]) ** 2
                )
                tv[r, c] += tmp.mean()
        tv /= height * width
        return tv

    patch_post = patch_TV(patch_post)
    img_post = resize(patch_post, preliminary_result.shape, order=0, mode='constant', anti_aliasing=False)
    return img_post
