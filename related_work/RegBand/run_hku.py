import json
import pickle
from os import makedirs, path

import gzip
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm, trange

from regular_band import RegularBand

if __name__ == "__main__":
    DATASET_DIR = "D:/Dataset/Textile/HKU/"
    RESULT_DIR = "./result/"
    makedirs(RESULT_DIR, exist_ok=True)
    IMG_SIZE = [256, 256]

    for P_NAME in ["box", "star", "dot"]:
        print("pattern ", P_NAME)
        rb = RegularBand()

        print("##### start training #####")
        train_names = glob(DATASET_DIR + P_NAME + "/normal/*.bmp")
        train_names = [path.basename(p) for p in train_names]
        img_base_path = DATASET_DIR + P_NAME + "/normal/%s"

        X = []
        for i in range(len(train_names)):
            name = train_names[i]
            good_img = Image.open(img_base_path % name).convert("L").resize(IMG_SIZE)
            good_img = np.array(good_img)
            X.append(good_img)
        X = np.asarray(X) / 255.0
        rb.fit(X)
        print("##### end training #####")

        print("##### start test #####")
        # names for test
        test_names = glob(DATASET_DIR + P_NAME + "/defective/*.bmp")
        test_names = [path.basename(p) for p in test_names]
        img_base_path = DATASET_DIR + P_NAME + "/defective/%s"
        X = []
        for i in range(len(test_names)):
            name = test_names[i]
            test_img = Image.open(img_base_path % name).convert("L").resize(IMG_SIZE)
            test_img = np.array(test_img)
            X.append(test_img)
        X = np.asarray(X) / 255.0
        mask_pred = rb.predict(X)
        print("##### end test #####")

        with gzip.open(RESULT_DIR + f"RB_pattern_{P_NAME}.pkl", "wb") as fp:
            pickle.dump({"names": test_names, "mask_pred": mask_pred}, fp)
