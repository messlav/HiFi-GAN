import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

from utils.utils import get_padding
from configs.train_batch_config import Config


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super(ResBlock, self).__init__()

        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, stride=1,
                padding=get_padding(kernel_size, dilation[0]), dilation=dilation[0]
            ))
            for dilation in dilations
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, stride=1,
                padding=get_padding(kernel_size, dilation[1]), dilation=dilation[1]
            ))
            for dilation in dilations
        ])

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_ = F.leaky_relu(x, 0.1)
            x_ = conv1(x_)
            x_ = F.leaky_relu(x_, 0.1)
            x_ = conv2(x_)
            x = x + x_

        return x


class MultiReceptiveField(nn.Module):
    def __init__(self, channels, kernels, dilations):
        super(MultiReceptiveField, self).__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(channels, kernel_size, dilation) for kernel_size, dilation in zip(kernels, dilations)
        ])

    def forward(self, x):
        out = self.resblocks[0](x)

        for resblock in self.resblocks[1:]:
            out += resblock(x)
        return out


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.first = nn.Conv1d(config.n_mels, config.n_channels, kernel_size=7, stride=1, padding=3)
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        chanels_now = config.n_channels
        for i in range(len(config.kernel_u)):
            out_channels = chanels_now // 2
            self.ups += [nn.ConvTranspose1d(chanels_now, out_channels, kernel_size=config.kernel_u[i],
                                            stride=config.upsample_size[i])]
            self.mrfs += [MultiReceptiveField(out_channels, config.kernel_r, config.dilations_r)]
            chanels_now = out_channels

        self.last = nn.Conv1d(chanels_now, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.first(x)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)

        x = F.leaky_relu(x, 0.1)
        x = self.last(x)
        x = torch.tanh(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.zeros_(m.bias)


def test():
    x = torch.randn((10, 80, 65))  # b x h x w
    G = Generator(Config)
    x = G(x)
    print(x.shape)
    print('done test')


if __name__ == '__main__':
    test()
