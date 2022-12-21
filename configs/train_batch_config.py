from dataclasses import dataclass
import torch


@dataclass
class Config:
    batch_size: int = 16
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4

    n_mels: int = 80
    n_channels: int = 512

    leaky: float = 0.1

    upsample_size = [8, 8, 2, 2]
    kernel_u = [16, 16, 4, 4]
    kernel_r = [3, 7, 11]
    dilations_r = [[[1, 1], [3, 1], [5, 1]]] * 3

    mpd_kernel_size: int = 5
    mpd_stride: int = 3
    mpd_periods = [2, 3, 5, 7, 11]

    mcd_num_layers = 3

    test_size: float = 0.1

    lambda_fm: float = 2.
    lambda_mel: float = 45

    wandb_project = 'Hi-Fi-GAN'

    log_train_step: int = 500
    log_val_step: int = 100

    n_epochs: int = 10

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
