from abc import abstractmethod

from tensorboardX import SummaryWriter
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .core_model import _CoreModel


class _TorchCoreModel(_CoreModel):
    def train(self):
        for attr_name, attr in vars(self).items():
            if isinstance(attr, Module):
                attr.train()
            elif isinstance(attr, list):
                if all(isinstance(m, Module) for m in attr):
                    for m in attr: m.train()

    def eval(self):
        for attr_name, attr in vars(self).items():
            if isinstance(attr, Module):
                attr.eval()
            elif isinstance(attr, list):
                if all(isinstance(m, Module) for m in attr):
                    for m in attr: m.eval()

    def dump_core_model(self) -> dict:
        attr_dict = self.attr_dict
        state_dict = {}
        for name, attr in attr_dict.items():
            if isinstance(attr, (Module, Optimizer)):
                state = attr.state_dict()
            elif isinstance(attr, str):
                state = getattr(self, attr)
            elif isinstance(attr, list):
                if all(isinstance(m, (Module, Optimizer)) for m in attr):
                    state = [m.state_dict() for m in attr]
            else:
                raise NotImplementedError
            state_dict[name] = state
        return state_dict

    def load_core_model(self, state_dict: dict):
        attr_dict = self.attr_dict
        for name, attr in attr_dict.items():
            if isinstance(attr, (Module, Optimizer)):
                attr.load_state_dict(state_dict[name])
            elif isinstance(attr, str):
                setattr(self, attr, state_dict[name])
            elif isinstance(attr, list):
                if all(isinstance(m, (Module, Optimizer)) for m in attr):
                    for m, s in zip(attr, state_dict[name]):
                        m.load_state_dict(s)
            else:
                raise NotImplementedError

    def init_attr_dict(self):
        self.attr_dict = {
            'model': self.model,
            'optimizer': self.optimizer,
        }

    @abstractmethod
    def init_core_model(self):
        self.model = 'some_core_model'
        self.model = ['model1', 'model2']
        self.loss_func = 'some_func'
        self.loss_func = ['func1', 'func2']
        self.optimizer = 'some_optimizer'
        self.optimizer = ['optimizer1', 'optimizer2']

    @abstractmethod
    def batch_fit(self, batch, epoch, **kwargs) -> dict:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        pred = self.model(images)
        model1, model2 = self.model
        preds1 = model1(images)
        preds2 = model2(images)
        loss = self.loss_func(preds1, labels)
        func1, func2 = self.loss_func
        loss1 = func1(preds1, labels)
        loss2 = func2(preds2, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        optim1, optim2 = self.optimizer
        optim1.zero_grad()
        loss1.backward()
        optim1.step()
        optim2.zero_grad()
        loss2.backward()
        optim2.step()
        losses = dict(loss1=loss1.item(), loss2=loss2.item(), loss=loss.item())
        return losses

    @abstractmethod
    def batch_predict(self, batch, **kwargs) -> dict:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        preds = self.model(images)
        results = dict(
            images=images.cpu().numpy(),
            labels=labels.cpu().numpy(),
            preds=preds.cpu().numpy(),
        )
        return results

    def after_one_epoch(self, writer: SummaryWriter, epoch: int, losses: dict, metrics: dict, **kwargs):
        pass
