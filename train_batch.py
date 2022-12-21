import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
import itertools
from tqdm import tqdm
import os

from utils.utils import set_random_seed, set_require_grad
from datasets.ljspeech_dataset import LJSpeechDataset, collate_fn
from melspec.melspec import MelSpectrogram, MelSpectrogramConfig
from configs.train_batch_config import Config
from model.generator import Generator
from model.discriminator import Discriminator
# from loss.gan_loss import FeatureLoss, DicriminatorLoss, GeneratorLoss
from loss.gan_loss import GanLoss, FeatureLoss, L1Loss
from utils.wandb_writer import WanDBWriter


def main():
    # configs
    train_config = Config()
    melspec_config = MelSpectrogramConfig()
    set_random_seed(3407)
    # data
    dataset = LJSpeechDataset('./')
    data_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    melspectrogram = MelSpectrogram(melspec_config).to(train_config.device)
    # model
    G = Generator(train_config)
    G = G.to(train_config.device)
    D = Discriminator(train_config)
    D = D.to(train_config.device)
    # loss, optimizer and hyperparameters
    # floss, gloss, dloss = FeatureLoss(), GeneratorLoss(), DicriminatorLoss()
    gan_loss = GanLoss()
    f_loss = FeatureLoss()
    l1_loss = L1Loss(melspectrogram, melspec_config.pad_value)
    optim_G = AdamW(G.parameters(), train_config.learning_rate, betas=(train_config.adam_b1, train_config.adam_b2))
    optim_D = AdamW(itertools.chain(D.msd.parameters(), D.mpd.parameters()),
                    train_config.learning_rate, betas=(train_config.adam_b1, train_config.adam_b2))
    scheduler_G = ExponentialLR(optim_G, gamma=train_config.lr_decay)
    scheduler_D = ExponentialLR(optim_D, gamma=train_config.lr_decay)
    scaler_G = GradScaler()
    scaler_D = GradScaler()
    os.makedirs(train_config.save_path, exist_ok=True)
    logger = WanDBWriter(train_config)
    logger.watch_model(D)
    logger.watch_model(G)  # not sure if that's working
    current_step = 0
    # train
    G.train()
    D.train()
    # tqdm_bar = tqdm(total=train_config.num_epochs * len(data_loader) - current_step)
    tqdm_bar = tqdm(total=100)
    waveforms, waveforms_length = next(iter(data_loader))
    waveforms = waveforms.to(train_config.device)
    melspec = melspectrogram(waveforms)
    # melspec = melspec.to(train_config.device)
    for i in range(100):
        current_step += 1
        tqdm_bar.update(1)
        logger.set_step(current_step)

        # Discriminator
        # TODO: autocast
        # with torch.no_grad():
        waveform_prediction = G(melspec)

        set_require_grad(D, True)
        set_require_grad(G, False)

        optim_D.zero_grad()

        mpd_fake, mpd_fake_feature_map = D.mpd(waveform_prediction.detach())
        msd_fake, msd_fake_feature_map = D.msd(waveform_prediction.detach())

        mpd_true, mpd_true_feature_map = D.mpd(waveforms.unsqueeze(1))
        msd_true, msd_true_feature_map = D.msd(waveforms.unsqueeze(1))

        D_real_loss = gan_loss(msd_true, mpd_true, fake=False)
        D_fake_loss = gan_loss(msd_fake, mpd_fake, fake=True)
        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        optim_D.step()

        logger.add_scalar("discriminator_fake_loss", D_fake_loss.detach().cpu().numpy())
        logger.add_scalar("discriminator_real_loss", D_real_loss.detach().cpu().numpy())
        logger.add_scalar("discriminator_loss", D_loss.detach().cpu().numpy())

        # Generator
        set_require_grad(D, False)
        set_require_grad(G, True)
        optim_G.zero_grad()
        waveform_l1 = l1_loss(melspec.unsqueeze(1), waveform_prediction) * train_config.lambda_mel

        mpd_fake, mpd_fake_feature_map = D.mpd(waveform_prediction)
        msd_fake, msd_fake_feature_map = D.msd(waveform_prediction)

        diff_len = waveform_prediction.shape[-1] - waveforms.shape[-1]
        waveform = F.pad(waveforms, (0, diff_len))

        mpd_true, mpd_true_feature_map = D.mpd(waveform.unsqueeze(1))
        msd_true, msd_true_feature_map = D.msd(waveform.unsqueeze(1))

        fm_loss = (f_loss(mpd_fake_feature_map, mpd_true_feature_map) +
                   f_loss(msd_fake_feature_map, msd_true_feature_map)) * train_config.lambda_fm

        gan_loss_now = gan_loss(msd_fake, mpd_fake, fake=False)

        G_loss = gan_loss_now + waveform_l1 + fm_loss
        G_loss.backward()
        optim_G.step()

        logger.add_scalar("gan_loss", gan_loss_now.detach().cpu().numpy())
        logger.add_scalar("l1_loss", waveform_l1.detach().cpu().numpy())
        logger.add_scalar("G_fm_loss", fm_loss.detach().cpu().numpy())
        logger.add_scalar("generator_loss", G_loss.detach().cpu().numpy())
        logger.add_scalar("G_learning_rate", scheduler_G.get_last_lr()[0])
        logger.add_scalar("D_learning_rate", scheduler_D.get_last_lr()[0])

        scheduler_G.step()
        scheduler_D.step()

    # TODO: save lr
    # TODO: Save G
    torch.save({'Generator': G.state_dict(), 'optimizer': optim_G.state_dict()},
               os.path.join(train_config.save_path, 'checkpoint_check.pth.tar'))

    G.eval()
    with torch.no_grad():
        fake = G(melspec)

    for i in range(3):
        logger.add_audio("true_audio", waveforms[i], sample_rate=22050)
        logger.add_audio("pred_audio", fake[i], sample_rate=22050)


if __name__ == '__main__':
    main()
