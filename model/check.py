from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm
import numpy as np
import torch

from configs.train_batch_config import Config


class MPDLayer(nn.Module):
    def __init__(self, period, kernel_size, stride, leaky, norm):
        super().__init__()
        self.period = period
        self.leaky = leaky

        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1, 32, kernel_size=(kernel_size, 1), stride=(stride, 1))),
            norm(nn.Conv2d(32, 64, kernel_size=(kernel_size, 1), stride=(stride, 1))),
            norm(nn.Conv2d(64, 128, kernel_size=(kernel_size, 1), stride=(stride, 1))),
            norm(nn.Conv2d(128, 256, kernel_size=(kernel_size, 1), stride=(stride, 1))),
            norm(nn.Conv2d(256, 1024, kernel_size=(kernel_size, 1), stride=(stride, 1)))
        ])

        self.postnet = nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1))

    def forward(self, x):
        batch_size, channels, seq_len = x.shape

        padded_len = int(np.ceil(seq_len / self.period) * self.period)
        if padded_len != seq_len:
            x = F.pad(x, (0, padded_len - seq_len))

        x = x.reshape(batch_size, channels, padded_len // self.period, self.period)

        feature_maps = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.leaky)

            feature_maps.append(x)

        return self.postnet(x), feature_maps


class MCDLayer(nn.Module):
    def __init__(self, norm, leaky):
        super().__init__()

        self.leaky = leaky
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
            norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=7)),
            norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=7)),
            norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=64, padding=7)),
            norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=256, padding=7)),
            norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=7))
        ])

        self.postnet = nn.Conv1d(1024, 1, kernel_size=3, stride=1)

    def forward(self, x):
        feature_maps = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.leaky)

            feature_maps.append(x)

        return self.postnet(x), feature_maps


class MPD(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.mpd_layers = nn.ModuleList([
            MPDLayer(period, config.mpd_kernel_size, config.mpd_stride, config.leaky, weight_norm)
            for period in config.mpd_periods
        ])

    def forward(self, x):
        feature_maps = []
        discriminator_scores = []

        for mpd in self.mpd_layers:
            pred, fmap = mpd(x)
            feature_maps.append(fmap)
            discriminator_scores.append(pred)

        return discriminator_scores, feature_maps


class MCD(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mcd_layers = nn.ModuleList([
            MCDLayer(weight_norm, config.leaky) for i in range(config.mcd_num_layers)
        ])

        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, 2),
            nn.AvgPool1d(4, 2, 2)
        ])

    def forward(self, x):
        feature_maps = []
        discriminator_scores = []

        for i in range(len(self.mcd_layers)):
            if i > 0:
                x = self.pools[i-1](x)

            pred, fmap = self.mcd_layers[i](x)
            feature_maps.append(fmap)
            discriminator_scores.append(pred)

        return discriminator_scores, feature_maps


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mcd = MCD(config)
        self.mpd = MPD(config)


def test():
    x = torch.randn((10, 1, 20000))  # b x nc x l
    D = Discriminator(Config)

    scores1, fmaps1 = D.mpd(x)
    print(len(scores1), len(fmaps1))
    print(len(scores1[0]), len(fmaps1[0]))
    print(scores1[0][0].shape, fmaps1[0][0].shape)
    print('done MPD\n')

    scores2, fmaps2 = D.mcd(x)
    print(len(scores2), len(fmaps2))
    print(len(scores2[0]), len(fmaps2[0]))
    print(scores2[0][0].shape, fmaps2[0][0].shape)
    print('done MSD')

    print('done test')


if __name__ == '__main__':
    test()
