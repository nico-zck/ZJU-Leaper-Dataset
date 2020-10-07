import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

from baselines.libs.configurer import Configurer
from baselines.libs.core_model import _SKLCoreModel
from baselines.libs.dataset.data_utils import merge_patches_to_image


class SparseCoder(_SKLCoreModel):
    def __init__(self, cfg: Configurer):
        super().__init__(cfg)
        cfg_hparam = cfg.HPARAM
        self.n_components = cfg_hparam.n_components
        self.alpha = cfg_hparam.alpha
        self.n_iter = cfg_hparam.n_iter
        self.batch_size = cfg_hparam.batch_size
        self.n_jobs = cfg_hparam.n_jobs
        self.init_core_model()

    def dump_core_model(self) -> dict:
        components = self.model.components_
        state_dict = dict(
            components=components,
        )
        return state_dict

    def load_core_model(self, state_dict: dict):
        components = state_dict['components']
        self.model.components_ = components

    def init_core_model(self):
        if hasattr(self, 'model'):
            del self.model
            import gc
            gc.collect()
        self.model = MiniBatchDictionaryLearning(
            n_components=self.n_components, alpha=self.alpha, n_jobs=16,
            n_iter=self.n_iter, batch_size=self.batch_size,
            fit_algorithm='lars', transform_algorithm='omp', verbose=True)

    def fit(self, train_data: list, **kwargs):
        patch_data = np.concatenate(train_data)
        patch_data = patch_data.reshape([np.prod(patch_data.shape[0:3]), np.prod(patch_data.shape[3:6])])
        self.model.fit(X=patch_data)

    def batch_predict(self, batch_data, **kwargs):
        name, label, image, patch, mask = batch_data
        shape = patch.shape
        patch = patch.reshape([np.prod(shape[0:3]), np.prod(shape[3:6])])
        code = self.model.transform(patch)
        patch_pred = np.dot(code, self.model.components_)
        patch_pred = patch_pred.reshape(shape)
        image_pred = []
        for img_patch in patch_pred:
            img_merged = merge_patches_to_image(patches=img_patch, patch_stride=self.cfg.DATASET.patch_stride)
            image_pred.append(img_merged)
        image_pred = np.asarray(image_pred)
        recon_h, recon_w = image_pred.shape[1:3]
        image = image[:, :recon_h, :recon_w]
        mask = mask[:, :recon_h, :recon_w]
        results = dict(
            name=name,
            label=label,
            image=image,
            mask=mask,
            image_pred=image_pred,
        )
        return results
