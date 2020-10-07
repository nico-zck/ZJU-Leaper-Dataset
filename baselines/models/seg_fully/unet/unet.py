import sys

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim

sys.path.append('/home/nico/PycharmProjects/defect-detection')
from baselines.libs.core_model import _TorchCoreModel
from baselines.libs.utils import network_init_kaiming


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout: layers.append(nn.Dropout())
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.pool(enc)
        return enc, x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_type='transpose', batch_norm=True):
        super(_DecoderBlock, self).__init__()
        if up_type == 'transpose':
            self.up_sample = nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=4, stride=2, padding=1)
        elif up_type == 'upsample':
            self.up_sample = nn.Upsample(scale_factor=2)
        else:
            raise NotImplementedError
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*layers)

    def forward(self, top, bottom):
        bottom = self.up_sample(bottom)
        return self.decoder(torch.cat([top, bottom], dim=1))


class _UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, base_filters=32):
        super().__init__()
        self.enc1 = _EncoderBlock(in_channels, base_filters)
        self.enc2 = _EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = _EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = _EncoderBlock(base_filters * 4, base_filters * 8, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 16, base_filters * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 16),
            nn.ReLU(inplace=True),
        )
        self.dec4 = _DecoderBlock(base_filters * 16, base_filters * 8)
        self.dec3 = _DecoderBlock(base_filters * 8, base_filters * 4)
        self.dec2 = _DecoderBlock(base_filters * 4, base_filters * 2)
        self.dec1 = _DecoderBlock(base_filters * 2, base_filters)
        self.final = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        enc1, x = self.enc1(x)
        enc2, x = self.enc2(x)
        enc3, x = self.enc3(x)
        enc4, x = self.enc4(x)
        center = self.center(x)
        dec4 = self.dec4(top=enc4, bottom=center)
        dec3 = self.dec3(top=enc3, bottom=dec4)
        dec2 = self.dec2(top=enc2, bottom=dec3)
        dec1 = self.dec1(top=enc1, bottom=dec2)
        final = self.final(dec1)
        return final


class UNet(_TorchCoreModel):
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
        self.lr = cfg_hparam['lr']
        self.weight_decay = cfg_hparam['weight_decay']
        self.init_core_model()
        self.init_attr_dict()

    def init_core_model(self):
        unet = _UNet(num_classes=1, in_channels=self.input_nc, base_filters=self.base_filters)
        unet.apply(network_init_kaiming())
        self.model = unet.cuda()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10,
                                                              cooldown=30, min_lr=1e-5, verbose=True)
        self.loss_func = nn.BCEWithLogitsLoss()
        print('{0:*^64}'.format(self.__class__.__name__))
        print(self.model)
        print('{0:*^64}'.format(''))

    def batch_fit(self, batch, epoch, **kwargs):
        name, label, img, mask = batch
        x_train = img.cuda()
        y_train = mask.cuda()
        self.optimizer.zero_grad()
        pred = self.model(x_train)
        loss = self.loss_func(pred, y_train)
        loss.backward()
        self.optimizer.step()
        losses = {"loss": loss.item()}
        return losses

    def batch_predict(self, batch, **kwargs):
        name, label, img, mask = batch
        raw_pred = self.model(img.cuda())
        raw_pred = torch.sigmoid(raw_pred)
        results = dict(
            name=name,
            label=label,
            image=img.cpu().numpy(),
            mask=mask.cpu().numpy(),
            raw_pred=raw_pred.cpu().numpy(),
        )
        return results

    def after_one_epoch(self, writer: SummaryWriter, epoch: int, losses: dict, metrics: dict, **kwargs):
        loss = losses["loss"]
        self.scheduler.step(loss)
