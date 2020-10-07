from os.path import join

import numpy as np
import torch
from thundersvm import OneClassSVM
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.libs.configurer import Configurer
from baselines.libs.core_model import _SKLCoreModel
from baselines.libs.dataset.data_utils import merge_patches_to_image
from baselines.models.seg_one_class.cae.cae import _CAE


class FeatCAE(_CAE):
    def __init__(self, in_channel, patch_size, fabric):
        super().__init__(in_channel, patch_size)
        pth_path = join(
            f'/home/nico/PycharmProjects/defect-detection/defect_detection/saved_models/CAE-201112/FabricFinal-{fabric}-cae/checkpoint-last.pth')
        checkpoint = torch.load(pth_path)
        print(checkpoint.keys())
        self.load_state_dict(checkpoint['core_model']['model'])

    def forward(self, x):
        x, _ = self.encoder_conv1(x)
        x, _ = self.encoder_conv2(x)
        x, _ = self.encoder_conv3(x)
        x, _ = self.encoder_conv4(x)
        return x


class OCSVMFeat(_SKLCoreModel):
    def __init__(self, cfg: Configurer):
        super().__init__(cfg)
        cfg_hparam = cfg.HPARAM
        self.kernel = cfg_hparam.kernel
        cfg_dataset = cfg.DATASET
        self.input_nc = 3 if cfg_dataset.color_mode == 'rgb' else 1
        self.patch_size = cfg_dataset.patch_size
        self.fabric = cfg_dataset.fabric
        self.init_core_model()

    def dump_core_model(self) -> dict:
        model_path = self.cfg.TEMP.model_dir + '/svm.model'
        self.model.save_to_file(model_path)
        return {}

    def load_core_model(self, state_dict: dict):
        model_path = self.cfg.TEMP.model_dir + '/svm.model'
        self.model.load_from_file(model_path)

    def init_core_model(self):
        if hasattr(self, 'model'):
            del self.model
            import gc
            gc.collect()
        self.model = OneClassSVM(
            kernel=self.kernel, verbose=True, n_jobs=16,
            gamma='auto',
        )
        self.feat_cae = FeatCAE(self.input_nc, self.patch_size, self.fabric).cuda()

    def fit(self, train_data: list, **kwargs):
        self.model.verbose = True
        train_data = torch.cat(train_data)
        shape = train_data.shape
        train_data = train_data.reshape([np.prod(shape[0:3]), *shape[3:6]])
        dl = DataLoader(train_data, batch_size=64)
        feats = []
        for patch_data in tqdm(dl, desc="getting features from CNN"):
            with torch.no_grad():
                batch_feat = self.feat_cae(patch_data.cuda())
                batch_feat = batch_feat.cpu().numpy().astype(np.float32)
                feats.append(batch_feat)
        feats = np.concatenate(feats)
        n_sample = feats.shape[0]
        n_feature = np.prod(feats.shape[1:4])
        feats = feats.reshape([n_sample, n_feature])
        self.model.fit(X=feats)

    def batch_predict(self, batch_data, **kwargs):
        self.model.verbose = False
        name, label, image, patch, mask = batch_data
        n_img, row, col, C, H, W = patch.shape
        patch = patch.reshape([n_img * row * col, C, H, W])
        with torch.no_grad():
            feats = self.feat_cae(patch.cuda())
            feats = feats.cpu().numpy().astype(np.float32)
        n_sample = feats.shape[0]
        n_feature = np.prod(feats.shape[1:4])
        feats = feats.reshape([n_sample, n_feature])
        label_pred = self.model.predict(feats).astype(np.int8)
        label_pred[label_pred == 1] = 0
        label_pred[label_pred == -1] = 1
        n_pixel = C * H * W
        patch_pred = label_pred[:, None] * np.ones([n_sample, n_pixel], dtype=np.int8)
        patch_pred = patch_pred.reshape([n_img, row, col, C, H, W])
        mask_pred = []
        for mask_patch in patch_pred:
            mask_merged = merge_patches_to_image(patches=mask_patch, patch_stride=self.cfg.DATASET.patch_stride)
            mask_pred.append(mask_merged)
        mask_pred = np.asarray(mask_pred)
        recon_h, recon_w = mask_pred.shape[1:3]
        image = image[:, :recon_h, :recon_w]
        mask = mask[:, :recon_h, :recon_w]
        results = dict(
            name=name,
            label=label,
            image=image,
            mask=mask,
            mask_pred=mask_pred,
        )
        return results
