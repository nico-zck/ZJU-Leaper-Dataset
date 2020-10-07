import torch.nn as nn
from torch import Tensor


def network_weights_regularization(net, p=2):
    assert isinstance(net, nn.Module)
    l2_reg_loss = 0
    for name, weight in net.named_parameters():
        if 'bias' not in name:
            l2_reg_loss += weight.norm(p=p)
    return l2_reg_loss


def network_init_xavier(nonlinearity='relu'):
    def init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain(nonlinearity))

    return init


def network_init_normal_distribution():
    def init(m):
        raise NotImplementedError

    return init


def network_init_kaiming(nonlinearity='relu'):
    def init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity=nonlinearity)

    return init


def compute_padding_for_pytorch(H_in, H_out, k_size, stride):
    padding = ((H_out - 1) * stride - H_in + k_size) // 2
    return padding


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Sequential(nn.Sequential):
    def forward(self, input):
        indices = None
        for module in self._modules.values():
            if isinstance(module, nn.MaxPool2d):
                if module.return_indices:
                    input, indices = module(input)
                else:
                    input = module(input)
            else:
                input = module(input)
        return input, indices


class SequentialTranspose(nn.Sequential):
    def forward(self, input):
        input, indices = input
        for module in self._modules.values():
            if isinstance(module, nn.MaxUnpool2d):
                if isinstance(indices, Tensor):
                    input = module(input, indices)
                else:
                    input = module(input)
            else:
                input = module(input)
        return input


def conv_pool(in_size, out_size, in_channels, out_channels, kernel_size,
              batch_norm=False, conv_stride=1, pool_size=2, index=True, activation='relu'):
    assert in_size // out_size == pool_size * conv_stride
    layer_list = []
    conv_out = in_size // conv_stride
    conv_padding = compute_padding_for_pytorch(in_size, conv_out, kernel_size, conv_stride)
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=conv_stride, padding=conv_padding)
    layer_list.append(conv_layer)
    if batch_norm:
        bn_layer = nn.BatchNorm2d(out_channels)
        layer_list.append(bn_layer)
    if activation == 'linear':
        activation_layer = nn.Identity()
    elif activation == 'tanh':
        activation_layer = nn.Tanh()
    elif activation == 'hard_tanh':
        activation_layer = nn.Hardtanh(inplace=True, min_val=-1, max_val=1)
    elif activation == 'leaky_relu':
        activation_layer = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    elif activation == 'prelu':
        activation_layer = nn.PReLU(num_parameters=1, init=0.25)
    elif activation == 'relu':
        activation_layer = nn.ReLU(inplace=True)
    else:
        activation_layer = nn.ReLU(inplace=True)
    layer_list.append(activation_layer)
    if pool_size > 1:
        pool_layer = nn.MaxPool2d(pool_size, stride=pool_size, return_indices=index)
        layer_list.append(pool_layer)
    return Sequential(*layer_list)


def conv_pool_transpose(in_size, out_size, in_channels, out_channels, kernel_size,
                        batch_norm=False, conv_stride=1, pool_size=2, activation='relu'):
    assert out_size // in_size == pool_size * conv_stride
    layer_list = []
    if pool_size > 1:
        in_size = in_size * pool_size
        pool_layer = nn.MaxUnpool2d(pool_size, stride=pool_size)
        layer_list.append(pool_layer)
    padding = compute_padding_for_pytorch(out_size, in_size, kernel_size, conv_stride)
    conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=conv_stride, padding=padding)
    layer_list.append(conv_layer)
    if batch_norm:
        bn_layer = nn.BatchNorm2d(out_channels)
        layer_list.append(bn_layer)
    if activation == 'linear':
        activation_layer = nn.Identity()
    elif activation == 'tanh':
        activation_layer = nn.Tanh()
    elif activation == 'hard_tanh':
        activation_layer = nn.Hardtanh(inplace=True, min_val=-1, max_val=1)
    elif activation == 'leaky_relu':
        activation_layer = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    elif activation == 'prelu':
        activation_layer = nn.PReLU(num_parameters=1, init=0.25)
    elif activation == 'relu':
        activation_layer = nn.ReLU(inplace=True)
    else:
        activation_layer = nn.ReLU(inplace=True)
    layer_list.append(activation_layer)
    return SequentialTranspose(*layer_list)


def make_layers(kernel_num_list, kernel_size_list, batch_norm=False):
    in_channels = 1
    layers = []
    for out_channels, kernel_size in zip(kernel_num_list, kernel_size_list):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = out_channels
    in_channels = kernel_num_list[-1]
    for out_channels, kernel_size in zip(kernel_num_list[::-1][0:-1], kernel_size_list[::-1][0:-1]):
        transpose_conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 2, kernel_size // 2, 1)
        if batch_norm:
            layers += [transpose_conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers += [transpose_conv2d, nn.ReLU(inplace=True)]
        in_channels = out_channels
    last_layer = nn.ConvTranspose2d(in_channels, 1, kernel_size_list[0], 2, kernel_size_list[0] // 2, 1)
    layers += [last_layer, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
