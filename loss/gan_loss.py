import torch.nn as nn
from torch.nn import functional as F


class GanLoss(nn.Module):
    def forward(self, msd_l, mpd_l, fake=True):
        loss = 0
        if fake:
            for s, p in zip(msd_l, mpd_l):
                loss += (s**2).mean() + (p**2).mean()
        else:
            for s, p in zip(msd_l, mpd_l):
                loss += ((s - 1)**2).mean() + ((p - 1)**2).mean()

        return loss


class L1Loss(nn.Module):
    def __init__(self, melspectrogram, pad_value):
        super(L1Loss, self).__init__()
        self.melspectrogram = melspectrogram
        self.loss = nn.L1Loss()
        self.pad_value = pad_value

    def forward(self, melspec, fake):
        melspec_fake = self.melspectrogram(fake)

        ln = melspec_fake.shape[-1] - melspec.shape[-1]
        if ln < 0:
            melspec_fake = F.pad(melspec_fake, (0, -ln), value=self.pad_value)
        else:
            melspec = F.pad(melspec, (0, ln), value=self.pad_value)

        return self.loss(melspec, melspec_fake)


class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, fake, real):
        loss = 0
        for fake_maps, real_maps in zip(fake, real):
            for fake_map, real_map in zip(fake_maps, real_maps):
                loss += self.loss(fake_map, real_map)

        return loss