import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import functional as F

from utils.utils import get_padding
from configs.train_batch_config import Config


class MultiPeriodDiscriminatorLayer(nn.Module):
    def __init__(self, period, kernel_size, stride, norm_f='weight'):
        super(MultiPeriodDiscriminatorLayer, self).__init__()
        self.period = period
        self.norm_f = spectral_norm if norm_f == 'spectral' else weight_norm
        self.convs = nn.ModuleList([
            self.norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            self.norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            self.norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            self.norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            self.norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.final = nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1))

    def forward(self, x):
        b, n_c, l = x.shape
        fmap = []
        padded_len = int(int(l / self.period) * self.period)
        if padded_len != l:
            x = F.pad(x, (0, padded_len - l))
        x = x.reshape(b, n_c, padded_len // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap += [x]

        x = self.final(x)
        fmap += [x]

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiPeriodDiscriminator, self).__init__()
        self.layers = nn.ModuleList([
            MultiPeriodDiscriminatorLayer(period, config.mpd_kernel_size, config.mpd_stride, weight_norm)
            for period in config.mpd_periods
        ])

    def forward(self, x):
        fmaps, scores = [], []
        for layer in self.layers:
            score, fmap = layer(x)
            scores += [score]
            fmaps += [fmap]

        return scores, fmaps


class MultiScaleDiscriminatorLayer(torch.nn.Module):
    def __init__(self, norm_f='weight'):
        super(MultiScaleDiscriminatorLayer, self).__init__()
        self.norm_f = spectral_norm if norm_f == 'spectral' else weight_norm
        self.convs = nn.ModuleList([
            self.norm_f(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
            self.norm_f(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20)),
            self.norm_f(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20)),
            self.norm_f(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
            self.norm_f(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
            self.norm_f(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20)),
            self.norm_f(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ])
        self.final = self.norm_f(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap += [x]

        x = self.final(x)
        fmap += [x]
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.layers = nn.ModuleList([
            MultiScaleDiscriminatorLayer(norm_f='weight'),
            MultiScaleDiscriminatorLayer(),
            MultiScaleDiscriminatorLayer(),
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
        ])

    def forward(self, x):
        fmaps, scores = [], []
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.pools[i - 1](x)
            pred, fmap = layer(x)
            fmaps += [fmap]
            scores += [pred]

        return scores, fmaps


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.mpd = MultiPeriodDiscriminator(config)
        self.msd = MultiScaleDiscriminator()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.zeros_(m.bias)


def test():
    x = torch.randn((10, 1, 20000))  # b x nc x l
    D = Discriminator(Config)

    scores1, fmaps1 = D.mpd(x)
    print(len(scores1), len(fmaps1))
    print(len(scores1[0]), len(fmaps1[0]))
    print(scores1[0][0].shape, fmaps1[0][0].shape)
    print('done MPD\n')

    scores2, fmaps2 = D.msd(x)
    print(len(scores2), len(fmaps2))
    print(len(scores2[0]), len(fmaps2[0]))
    print(scores2[0][0].shape, fmaps2[0][0].shape)
    print('done MSD')

    print('done test')


if __name__ == '__main__':
    test()
