import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim

from baselines.libs.core_model import _TorchCoreModel
from baselines.models.seg_fully.unet.unet import _UNet


class BaseUNet(_UNet):
    def __init__(self, num_classes, in_channels=3, base_filters=32):
        super().__init__(num_classes, in_channels, base_filters)
        self.final = nn.Identity()


class SharedUNet(nn.Module):
    def __init__(self, base_unet, num_classes, base_filters=32):
        super().__init__()
        assert isinstance(base_unet, BaseUNet)
        self.base_unet = base_unet
        self.output = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        return self.output(self.base_unet(x))


class UNetTransferUnsup(_TorchCoreModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        cfg_dataset = cfg.DATASET
        self.color_mode = cfg_dataset['color_mode']
        if self.color_mode == 'rgb':
            self.input_nc = 3
        else:
            self.input_nc = 1
        cfg_hparam = cfg.HPARAM
        self.base_filters = cfg_hparam['base_filters']
        self.weight_decay = cfg_hparam['weight_decay']
        self.lr = cfg_hparam['lr']
        self.epoch_s1, self.epoch_s2 = cfg_hparam['epochs']
        self.init_core_model()
        self.init_attr_dict()

    def init_core_model(self):
        model_base = BaseUNet(num_classes=1, in_channels=self.input_nc, base_filters=self.base_filters)
        model_s1 = SharedUNet(base_unet=model_base, num_classes=self.input_nc, base_filters=self.base_filters)
        model_s1 = model_s1.cuda()
        model_s2 = SharedUNet(base_unet=model_base, num_classes=1, base_filters=self.base_filters)
        model_s2 = model_s2.cuda()
        self.model = [model_base, model_s1, model_s2]
        optimizer_s1 = optim.Adam(params=model_s1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_s2 = optim.Adam(params=model_s2.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = [optimizer_s1, optimizer_s2]
        loss_fn_s1 = nn.MSELoss()
        loss_fn_s2 = nn.BCEWithLogitsLoss()
        self.loss_func = [loss_fn_s1, loss_fn_s2]

    def batch_fit(self, batch, epoch, **kwargs) -> dict:
        _, model_s1, model_s2 = self.model
        optimizer_s1, optimizer_s2 = self.optimizer
        loss_fn_s1, loss_fn_s2 = self.loss_func
        loss_s1 = loss_s2 = 0
        if epoch <= self.epoch_s1:
            name, label, img_src, img_tgt = batch
            x = img_src.cuda()
            y = img_tgt.cuda()
            optimizer_s1.zero_grad()
            p = model_s1(x)
            loss_s1 = loss_fn_s1(p, y)
            loss_s1.backward()
            optimizer_s1.step()
            loss_s1 = loss_s1.item()
        else:
            name, label, image, mask = batch
            x = image.cuda()
            y = mask.cuda()
            optimizer_s2.zero_grad()
            p = model_s2(x)
            loss_s2 = loss_fn_s2(p, y)
            loss_s2.backward()
            optimizer_s2.step()
            loss_s2 = loss_s2.item()
        losses = dict(loss_s1=loss_s1, loss_s2=loss_s2)
        return losses

    def after_one_epoch(self, writer: SummaryWriter, epoch: int, losses: dict, metrics: dict, **kwargs):
        if epoch == self.epoch_s1:
            print("freezing the weights when finished pre-training")
            model_base, _, _ = self.model
            fixed_modules = ['enc1', 'enc2', 'enc3']
            print('freezing weights for %s layers' % str(fixed_modules))
            for name, m in model_base.named_modules():
                if name in fixed_modules:
                    for param in m.parameters():
                        param.requires_grad = False
        if epoch == (self.epoch_s1 + self.epoch_s2 - 60):
            print("unfreezing model weights for fine-tuning!")
            model_base, _, _ = self.model
            for name, m in model_base.named_modules():
                for param in m.parameters():
                    param.requires_grad = True

    def batch_predict(self, batch, **kwargs) -> dict:
        name, label, image, mask = batch
        _, model_s1, model_s2 = self.model
        x = image.cuda()
        p = model_s2(x)
        rp = torch.sigmoid(p)
        results = dict(
            name=name,
            label=label,
            image=image.cpu().numpy(),
            mask=mask.cpu().numpy(),
            raw_pred=rp.cpu().numpy()
        )
        return results
