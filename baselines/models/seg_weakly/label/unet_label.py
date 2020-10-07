import sys

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim

sys.path.append('/home/nico/PycharmProjects/defect-detection')
from baselines.libs.utils import network_init_kaiming
from baselines.models.seg_fully.unet.unet import UNet, _UNet


class _UNetLabel(_UNet):
    @staticmethod
    def LogSumExp(x, r=5):
        N, C, ho, wo = x.shape
        a, _ = (r * x).view(N, C, -1).max(dim=2)
        aa = a[..., None, None]
        y_out = (1. / r) * \
                (a + torch.log((1. / (ho * wo)) * torch.sum(torch.exp(r * x - aa), dim=[2, 3])))
        return y_out

    def forward(self, x):
        final = super().forward(x)
        pred_mask = final
        pred_lse = self.LogSumExp(final).squeeze(dim=1)
        return pred_lse, pred_mask


class UNetLabel(UNet):
    def init_core_model(self):
        unet = _UNetLabel(num_classes=1, in_channels=self.input_nc, base_filters=self.base_filters)
        unet.apply(network_init_kaiming())
        self.model = unet.cuda()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.loss_func = nn.BCEWithLogitsLoss()
        print('{0:*^64}'.format(self.__class__.__name__))
        print(self.model)
        print('{0:*^64}'.format(''))

    def batch_fit(self, batch, epoch, **kwargs):
        name, label, img = batch
        x_img = img.cuda()
        y_label = label.float().cuda()
        self.optimizer.zero_grad()
        pred_lse, pred_mask = self.model(x_img)
        loss = self.loss_func(pred_lse, y_label)
        loss.backward()
        self.optimizer.step()
        losses = {"loss": loss.item() * 1e5}
        return losses

    def batch_predict(self, batch, **kwargs):
        name, label, img, mask = batch
        pred_lse, pred_mask = self.model(img.cuda())
        raw_pred = torch.sigmoid(pred_mask)
        results = dict(
            name=name,
            label=label,
            image=img.cpu().numpy(),
            mask=mask.cpu().numpy(),
            raw_pred=raw_pred.cpu().numpy(),
        )
        return results

    def after_one_epoch(self, writer: SummaryWriter, epoch: int, losses: dict, metrics: dict, **kwargs):
        self.scheduler.step()
