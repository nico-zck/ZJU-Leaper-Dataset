import json
import pickle
from os import makedirs

import gzip
import numpy as np
from PIL import Image
from easydict import EasyDict
from tqdm import tqdm, trange

from regular_band import RegularBand

if __name__ == "__main__":
    DATASET_DIR = "D:/Dataset/ZJU/Dataset-Fabric/FabricFinal/"
    # DATASET_DIR = 'D:/zck/Dataset/FabricFinal/'
    RESULT_DIR = "./result/"
    makedirs(RESULT_DIR, exist_ok=True)
    IMG_SIZE = [256, 256]

    for P_ID in range(1, 5):
        print("pattern ", P_ID)
        rb = RegularBand()

        print("##### start training #####")
        img_base_path = DATASET_DIR + "Images/%s.jpg"
        json_path = DATASET_DIR + f"ImageSets/Patterns/pattern{P_ID}.json"
        with open(json_path, "r") as fp:
            all_names = EasyDict(json.load(fp))
        # names for normal images
        train_names = all_names.normal.train

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
        test_names = all_names.normal.test + all_names.defect.test
        X = []
        for i in range(len(test_names)):
            name = test_names[i]
            test_img = Image.open(img_base_path % name).convert("L").resize(IMG_SIZE)
            test_img = np.array(test_img)
            X.append(test_img)
        X = np.asarray(X) / 255.0
        mask_pred = rb.predict(X)
        print("##### end test #####")

        with gzip.open(RESULT_DIR + f"RB_pattern_{P_ID}.pkl", "wb") as fp:
            pickle.dump({"names": test_names, "mask_pred": mask_pred}, fp)
