from dataclasses import dataclass
import torch


@dataclass
class Config:
    wandb_project: str = 'Hi-Fi-GAN'
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    num_epochs = 100
    save_epochs = 10
    validate_epochs = 1
    save_path = 'weights'

    batch_size: int = 16
    num_workers: int = 8

    n_mels: int = 80
    n_channels: int = 512

    upsample_size = [8, 8, 2, 2]
    kernel_u = [16, 16, 4, 4]
    kernel_r = [3, 7, 11]
    dilations_r = [[[1, 1], [3, 1], [5, 1]]] * 3
    mpd_kernel_size: int = 5
    mpd_stride: int = 3
    mpd_periods = [2, 3, 5, 7, 11]
    mcd_num_layers = 3
    lambda_fm: float = 2.
    lambda_mel: float = 45

    optim = 'AdamW'
    learning_rate: int = 0.0002
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999

    def get(self, attr, default_value=None):
        return getattr(self, attr, default_value)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)

        raise KeyError(key)
