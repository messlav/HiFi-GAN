import torch.nn as nn
import torch
from torch.nn import functional as F


# class FeatureLoss(nn.Module):
#     def forward(self, fmaps_real, fmaps_gen):
#         loss = 0
#         for fmap_real, fmap_gen in zip(fmaps_real, fmaps_gen):
#             for real, gen in zip(fmap_real, fmap_gen):
#                 loss += torch.mean(torch.abs(real - gen))
#
#         return 2 * loss
#
#
# class DicriminatorLoss(nn.Module):
#     def forward(self, real, fake):
#         loss = 0
#         for r, f in zip(real, fake):
#             loss += torch.mean((1 - r)**2) + torch.mean(f**2)
#
#         return loss
#
#
# class GeneratorLoss(nn.Module):
#     def forward(self, disc):
#         loss = 0
#         for d in disc:
#             loss += torch.mean((1 - d)**2)
#
#         return loss


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
    def __init__(self, melspectrogram):
        super(L1Loss, self).__init__()
        self.melspectrogram = melspectrogram
        self.loss = nn.L1Loss()

    def forward(self, melspec, waveform_prediction):
        melspec_prediction = self.melspectrogram(waveform_prediction)

        diff_len = melspec_prediction.shape[-1] - melspec.shape[-1]

        if diff_len < 0:
            melspec_prediction = F.pad(melspec_prediction, (0, -diff_len), value=self.pad_value)
        else:
            melspec = F.pad(melspec, (0, diff_len), value=self.pad_value)

        waveform_l1 = self.loss(
            melspec,
            melspec_prediction
        )

        return waveform_l1


class FeatureLoss(nn.Module):
    def forward(self, fake_fms_arr, true_fms_arr):
        return sum(
            sum([self.fm_loss(fake_fm, true_fm) for fake_fm, true_fm in zip(fake_fms, true_fms)]) for fake_fms, true_fms
            in zip(fake_fms_arr, true_fms_arr))
