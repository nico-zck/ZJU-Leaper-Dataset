import sys

import numpy as np
import torch.nn as nn
from torch import optim

sys.path.append('/home/nico/PycharmProjects/defect-detection')
from baselines.libs.core_model import _TorchCoreModel
from baselines.libs.dataset.data_utils import merge_patches_to_image
from baselines.libs.utils import network_init_kaiming, conv_pool, conv_pool_transpose


class _CAE(nn.Module):
    def __init__(self, in_channel, patch_size):
        super().__init__()
        kernel_num = [20, 40, 60, 80]
        kernel_size = [12, 12, 5, 5]
        hidden_size = [64, 32, 16, 8]
        conv_stride = [2, 2, 1, 1]
        pool_size = [1, 1, 2, 2]
        self.encoder_conv1 = conv_pool(patch_size, hidden_size[0], in_channel, kernel_num[0], kernel_size[0],
                                       conv_stride=conv_stride[0], pool_size=pool_size[0], index=True)
        self.encoder_conv2 = conv_pool(hidden_size[0], hidden_size[1], kernel_num[0], kernel_num[1], kernel_size[1],
                                       conv_stride=conv_stride[1], pool_size=pool_size[1], index=True)
        self.encoder_conv3 = conv_pool(hidden_size[1], hidden_size[2], kernel_num[1], kernel_num[2], kernel_size[2],
                                       conv_stride=conv_stride[2], pool_size=pool_size[2], index=True)
        self.encoder_conv4 = conv_pool(hidden_size[2], hidden_size[3], kernel_num[2], kernel_num[3], kernel_size[3],
                                       conv_stride=conv_stride[3], pool_size=pool_size[3], index=True)
        self.decoder_deconv4 = conv_pool_transpose(hidden_size[3], hidden_size[2], kernel_num[3], kernel_num[2],
                                                   kernel_size[3],
                                                   conv_stride=conv_stride[3], pool_size=pool_size[3])
        self.decoder_deconv3 = conv_pool_transpose(hidden_size[2], hidden_size[1], kernel_num[2], kernel_num[1],
                                                   kernel_size[2],
                                                   conv_stride=conv_stride[2], pool_size=pool_size[2])
        self.decoder_deconv2 = conv_pool_transpose(hidden_size[1], hidden_size[0], kernel_num[1], kernel_num[0],
                                                   kernel_size[1],
                                                   conv_stride=conv_stride[1], pool_size=pool_size[1])
        self.decoder_deconv1 = conv_pool_transpose(hidden_size[0], patch_size, kernel_num[0], in_channel,
                                                   kernel_size[0],
                                                   conv_stride=conv_stride[0], pool_size=pool_size[0],
                                                   activation='tanh')

    def forward(self, x):
        x, indices1 = self.encoder_conv1(x)
        x, indices2 = self.encoder_conv2(x)
        x, indices3 = self.encoder_conv3(x)
        x, indices4 = self.encoder_conv4(x)
        x = self.decoder_deconv4([x, indices4])
        x = self.decoder_deconv3([x, indices3])
        x = self.decoder_deconv2([x, indices2])
        x = self.decoder_deconv1([x, indices1])
        return x


class CAE(_TorchCoreModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        cfg_dataset = cfg.DATASET
        self.color_mode = cfg_dataset['color_mode']
        if self.color_mode == 'rgb':
            self.input_nc = 3
        else:
            self.input_nc = 1
        self.patch_size = cfg_dataset.patch_size
        self.patch_stride = cfg_dataset.patch_stride
        cfg_hparam = cfg.HPARAM
        self.lr = cfg_hparam['lr']
        self.weight_decay = cfg_hparam['weight_decay']
        self.init_core_model()
        self.init_attr_dict()

    def init_core_model(self):
        model = _CAE(self.input_nc, self.patch_size)
        model.apply(network_init_kaiming())
        self.model = model.cuda()
        self.optimizer = optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr, gamma=0.1)
        self.loss_func = nn.MSELoss()
        print('{0:*^64}'.format(self.__class__.__name__))
        print(self.model)
        print('{0:*^64}'.format(''))

    def batch_fit(self, batch, epoch, **kwargs) -> dict:
        patch_data = batch
        shape = patch_data.shape
        patch_data = patch_data.reshape([np.prod(shape[0:3]), *shape[3:6]])
        x = patch_data.cuda()
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_func(pred, x)
        loss.backward()
        self.optimizer.step()
        losses = dict(loss=loss.item())
        return losses

    def batch_predict(self, batch, **kwargs) -> dict:
        name, label, image, patch, mask = batch
        shape = patch.shape
        patch = patch.reshape([np.prod(shape[0:3]), *shape[3:6]])
        patch_pred = self.model(patch.cuda())
        patch_pred = patch_pred.cpu().numpy().reshape(shape)
        image_pred = []
        for img_patch in patch_pred:
            img_merged = merge_patches_to_image(patches=img_patch, patch_stride=self.patch_stride)
            image_pred.append(img_merged)
        image_pred = np.asarray(image_pred)
        recon_h, recon_w = image_pred.shape[1:3]
        image = image[:, :recon_h, :recon_w]
        mask = mask[:, :recon_h, :recon_w]
        results = dict(
            name=name,
            image=image,
            mask=mask,
            image_pred=image_pred,
        )
        return results
