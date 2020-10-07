import sys

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

sys.path.append('/home/nico/PycharmProjects/defect-detection')
from baselines.models.seg_fully.unet.unet import UNet


class UNetBBox(UNet):
    def init_core_model(self):
        super().init_core_model()
        self.scheduler = StepLR(self.optimizer, step_size=40, gamma=0.1)

    def batch_fit(self, batch, epoch, **kwargs):
        return super().batch_fit(batch, epoch, **kwargs)

    def batch_predict(self, batch, **kwargs):
        return super().batch_predict(batch, **kwargs)

    def after_one_epoch(self, writer: SummaryWriter, epoch: int, losses: dict, metrics: dict, **kwargs):
        self.scheduler.step()
