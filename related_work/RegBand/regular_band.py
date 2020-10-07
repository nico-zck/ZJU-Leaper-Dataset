# -*- coding: utf-8 -*-
"""
@Author: Nico
@Paper: Ngan, H. Y. T., and G. K. H. Pang. “Regularity Analysis for Patterned Texture Inspection.” IEEE Transactions on Automation Science and Engineering, vol. 6, no. 1, Jan. 2009
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import makedirs

import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_WORKER = 32


class RegularBand:
    def __init__(self, win_size=25):
        self.win_size = win_size

    def fit(self, X, **kwargs):
        X = X.squeeze()
        X = X - X.mean()

        samples = X.shape[0]
        row_mean_bounds = np.zeros([samples, 4])
        col_mean_bounds = np.zeros([samples, 4])

        """3 Calculation of the Regular bands"""
        with ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
            futures = [
                executor.submit(self.compute_LRB_DRB_bounds, img_data) for img_data in X
            ]
            row_mean_bounds = []
            col_mean_bounds = []
            for f in tqdm(futures, desc="training", dynamic_ncols=True):
                row_, col_ = f.result()
                row_mean_bounds.append(row_)
                col_mean_bounds.append(col_)

        """4 Obtain the threshold values"""
        row_mean_bounds = np.asarray(row_mean_bounds).mean(axis=1)
        col_mean_bounds = np.asarray(col_mean_bounds).mean(axis=1)

        # [LRB.min, LRB.max, DRB.min, DRB.max]
        self.row_thresholds = row_mean_bounds
        self.col_thresholds = col_mean_bounds

    def predict(self, X, **kwargs):
        X = X.squeeze()
        X = X - X.mean()

        with ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
            futures = [executor.submit(self.predict_image, img_data) for img_data in X]
            mask_pred = []
            for f in tqdm(futures, desc="testing", dynamic_ncols=True):
                mask_pred.append(f.result())

        mask_pred = np.asarray(mask_pred)
        return mask_pred

    def predict_image(self, img_data):
        """5 Calculate the Light and Dark Regular Bands on rows and columns of every image."""
        (
            row_lrb_img,
            row_drb_img,
            col_lrb_img,
            col_drb_img,
        ) = self.compute_LRB_DRB_image(img_data)

        # self.plot_3d(
        #     row_lrb_img, row_drb_img, col_lrb_img, col_drb_img, index,
        # )

        """6 Threshold the Regular Bands matrices with the corresponding threshold values determined during the training stage."""
        # Two Lemmas LRB DRB always grater than 0
        row_lrb_result = np.logical_or(
            row_lrb_img < self.row_thresholds[0], row_lrb_img > self.row_thresholds[1],
        )
        row_drb_result = np.logical_or(
            row_drb_img < self.row_thresholds[2], row_drb_img > self.row_thresholds[3],
        )
        col_lrb_result = np.logical_or(
            col_lrb_img < self.col_thresholds[0], col_lrb_img > self.col_thresholds[1],
        )
        col_drb_result = np.logical_or(
            col_drb_img < self.col_thresholds[2], col_drb_img > self.col_thresholds[3],
        )

        """8 Combine the thresholded results of the LRB and of the DRB, from the row of one image with an or-operation."""
        row_result = np.logical_or(row_lrb_result, row_drb_result)
        """9 Then, repeat the same step on the column side."""
        col_result = np.logical_or(col_lrb_result, col_drb_result)
        """7 Perform zero padding on the thresholded matrices"""
        row_padding = np.zeros_like(img_data, dtype=np.bool)
        row_padding[-row_result.shape[0]:, -row_result.shape[1]:] = row_result
        row_result = row_padding
        col_padding = np.zeros_like(img_data, dtype=np.bool)
        col_padding[-col_result.shape[0]:, -col_result.shape[1]:] = col_result
        col_result = col_padding

        """10 Combine the results of rows and columns as a whole with an or-operation"""
        binary_result = np.logical_or(row_result, col_result)
        return binary_result

    def compute_LRB_DRB_bounds(self, img_data):
        rows, cols = img_data.shape

        row_lrb_img, row_drb_img, col_lrb_img, col_drb_img = self.compute_LRB_DRB_image(
            img_data
        )

        # [LRB.min, LRB.max, DRB.min, DRB.max]
        row_bounds = np.zeros([rows, 4])
        col_bounds = np.zeros([cols, 4])
        for r in range(rows):
            r_lrb = row_lrb_img[r, :]
            r_drb = row_drb_img[r, :]
            # LRB, DRB bounds on rows
            row_bounds[r] = np.asarray(
                [r_lrb.min(), r_lrb.max(), r_drb.min(), r_drb.max()]
            )

        for c in range(cols):
            c_lrb = col_lrb_img[:, c]
            c_drb = col_drb_img[:, c]
            # LRB, DRB bounds on columns
            col_bounds[c] = np.asarray(
                [c_lrb.min(), c_lrb.max(), c_drb.min(), c_drb.max()]
            )

        # row_extremum_bound = row_bounds.mean(axis=0)
        # col_extremum_bound = col_bounds.mean(axis=0)
        row_extremum_bound = np.asarray(
            [
                row_bounds[:, 0].min(),
                row_bounds[:, 1].max(),
                row_bounds[:, 2].min(),
                row_bounds[:, 3].max(),
            ]
        )
        col_extremum_bound = np.asarray(
            [
                col_bounds[:, 0].min(),
                col_bounds[:, 1].max(),
                col_bounds[:, 2].min(),
                col_bounds[:, 3].max(),
            ]
        )

        return row_extremum_bound, col_extremum_bound

    def compute_LRB_DRB_image(self, img_data):
        rows, cols = img_data.shape

        row_lrb_img = []
        row_drb_img = []
        col_lrb_img = []
        col_drb_img = []

        for r in range(rows):
            r_mean, r_std = self.moving_average_and_variance(
                img_data[r, :], self.win_size
            )
            # LRB, DRB on rows
            row_lrb_img.append(np.abs(r_mean - r_std) + r_mean)
            row_drb_img.append(np.abs(r_mean + r_std) - r_mean)

        for c in range(cols):
            # LRB, DRB on columns
            c_mean, c_std = self.moving_average_and_variance(
                img_data[:, c], self.win_size
            )
            col_lrb_img.append(np.abs(c_mean - c_std) + c_mean)
            col_drb_img.append(np.abs(c_mean + c_std) - c_mean)

        row_lrb_img = np.stack(row_lrb_img)
        row_drb_img = np.stack(row_drb_img)
        col_lrb_img = np.column_stack(col_lrb_img)
        col_drb_img = np.column_stack(col_drb_img)
        return row_lrb_img, row_drb_img, col_lrb_img, col_drb_img

    @staticmethod
    def moving_average_and_variance(rolling_data, win_size):
        rolling_data = pd.Series(rolling_data).rolling(win_size)

        mean = rolling_data.mean()
        std = rolling_data.std()

        mean = mean.dropna()
        std = std.dropna()

        # mean = np.nan_to_num(mean)
        # std = np.nan_to_num(std)
        return mean, std

    def plot_3d(self, row_lrb_img, row_drb_img, col_lrb_img, col_drb_img, index):
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        nums, length = row_drb_img.shape
        xx, yy = np.meshgrid([0, 256], [0, 256])
        zz = np.ones([2, 2])

        fig = plt.figure(figsize=[15, 10])
        cp = sns.cubehelix_palette(n_colors=256, rot=-0.4)

        ax1 = fig.add_subplot(221, projection=Axes3D.name)
        ax1.plot_surface(xx, yy, zz * self.row_thresholds[0], alpha=0.6)
        ax1.plot_surface(xx, yy, zz * self.row_thresholds[1], alpha=0.6)
        y = np.arange(length)
        for r in range(nums):
            x = np.ones_like(y) * r
            ax1.plot(x, y, row_lrb_img[r, :], color=cp[r])
        ax1.set_title("row LRB")

        ax2 = fig.add_subplot(222, projection=Axes3D.name)
        ax2.plot_surface(xx, yy, zz * self.row_thresholds[2], alpha=0.6)
        ax2.plot_surface(xx, yy, zz * self.row_thresholds[3], alpha=0.6)
        y = np.arange(length)
        for r in range(nums):
            x = np.ones_like(y) * r
            ax2.plot(x, y, row_drb_img[r, :], color=cp[r])
        ax2.set_title("row LDB")

        ax3 = fig.add_subplot(223, projection=Axes3D.name)
        ax3.plot_surface(xx, yy, zz * self.col_thresholds[0], alpha=0.6)
        ax3.plot_surface(xx, yy, zz * self.col_thresholds[1], alpha=0.6)
        x = np.arange(length)
        for c in range(nums - 1, -1, -1):
            y = np.ones_like(x) * c
            ax3.plot(x, y, col_lrb_img[:, c], color=cp[c])
        ax3.set_title("col LRB")

        ax4 = fig.add_subplot(224, projection=Axes3D.name)
        ax4.plot_surface(xx, yy, zz * self.col_thresholds[2], alpha=0.6)
        ax4.plot_surface(xx, yy, zz * self.col_thresholds[3], alpha=0.6)
        x = np.arange(length)
        for c in range(nums - 1, -1, -1):
            y = np.ones_like(x) * c
            ax4.plot(x, y, col_drb_img[:, c], color=cp[c])
        ax4.set_title("col LDB")

        makedirs("./display/", exist_ok=True)
        plt.savefig("./display/img%d_3d.png" % index)
        plt.close()
