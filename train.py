import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
import itertools
from tqdm import tqdm
import os

from utils.utils import set_random_seed, set_require_grad
from datasets.ljspeech_dataset import LJSpeechDataset, collate_fn
from datasets.test_dataset import TestDataset
from melspec.melspec import MelSpectrogram, MelSpectrogramConfig
from configs.train_config import Config
from model.generator import Generator
from model.discriminator import Discriminator
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
    testset = TestDataset('val/wavs')
    test_loader = DataLoader(
        testset,
        batch_size=3,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    melspectrogram = MelSpectrogram(melspec_config).to(train_config.device)
    waveforms_test, waveforms_length_test = next(iter(test_loader))
    waveforms_test = waveforms_test.to(train_config.device)
    melspec_test = melspectrogram(waveforms_test)
    # model
    G = Generator(train_config)
    G = G.to(train_config.device)
    D = Discriminator(train_config)
    D = D.to(train_config.device)
    # loss, optimizer and hyperparameters
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
    tqdm_bar = tqdm(total=train_config.num_epochs * len(data_loader) - current_step)
    for epoch in range(train_config.num_epochs):
        for waveforms, waveforms_length in data_loader:
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)

            waveforms = waveforms.to(train_config.device)
            melspec = melspectrogram(waveforms)

            # Discriminator
            with torch.cuda.amp.autocast():
                waveform_prediction = G(melspec)

                set_require_grad(D, True)
                set_require_grad(G, False)

                mpd_fake, mpd_fake_fmap = D.mpd(waveform_prediction.detach())
                msd_fake, msd_fake_fmap = D.msd(waveform_prediction.detach())

                mpd_real, mpd_real_fmap = D.mpd(waveforms.unsqueeze(1))
                msd_real, msd_real_fmap = D.msd(waveforms.unsqueeze(1))

                D_real_loss = gan_loss(msd_real, mpd_real, fake=False)
                D_fake_loss = gan_loss(msd_fake, mpd_fake, fake=True)
                D_loss = D_real_loss + D_fake_loss

            optim_D.zero_grad()
            scaler_D.scale(D_loss).backward()
            scaler_D.step(optim_D)
            scaler_D.update()

            logger.add_scalar("discriminator_fake_loss", D_fake_loss.detach().cpu().numpy())
            logger.add_scalar("discriminator_real_loss", D_real_loss.detach().cpu().numpy())
            logger.add_scalar("discriminator_loss", D_loss.detach().cpu().numpy())

            # Generator
            set_require_grad(D, False)
            set_require_grad(G, True)
            with torch.cuda.amp.autocast():
                l1_loss_now = l1_loss(melspec, waveform_prediction.squeeze(1)) * train_config.lambda_mel

                mpd_fake, mpd_fake_fmap = D.mpd(waveform_prediction)
                msd_fake, msd_fake_fmap = D.msd(waveform_prediction)

                ln = waveform_prediction.shape[-1] - waveforms.shape[-1]
                waveform = F.pad(waveforms, (0, ln))

                mpd_real, mpd_real_fmap = D.mpd(waveform.unsqueeze(1))
                msd_real, msd_real_fmap = D.msd(waveform.unsqueeze(1))

                fm_loss = (f_loss(mpd_fake_fmap, mpd_real_fmap) +
                           f_loss(msd_fake_fmap, msd_real_fmap)) * train_config.lambda_fm

                gan_loss_now = gan_loss(msd_fake, mpd_fake, fake=False)

                G_loss = gan_loss_now + l1_loss_now + fm_loss

            optim_G.zero_grad()
            scaler_G.scale(G_loss).backward()
            scaler_G.step(optim_G)
            scaler_G.update()

            logger.add_scalar("gan_loss", gan_loss_now.detach().cpu().numpy())
            logger.add_scalar("l1_loss", l1_loss_now.detach().cpu().numpy())
            logger.add_scalar("G_fm_loss", fm_loss.detach().cpu().numpy())
            logger.add_scalar("generator_loss", G_loss.detach().cpu().numpy())
            logger.add_scalar("G_learning_rate", scheduler_G.get_last_lr()[0])
            logger.add_scalar("D_learning_rate", scheduler_D.get_last_lr()[0])

        scheduler_G.step()
        scheduler_D.step()

        if epoch != 0 and epoch % train_config.save_epochs == 0:
            torch.save({'Generator': G.state_dict(), 'optimizer': optim_G.state_dict()},
                       os.path.join(train_config.save_path, 'checkpoint_%d.pth.tar' % epoch))

        if epoch % train_config.validate_epochs == 0 or epoch == train_config.num_epochs - 1:
            G.eval()
            with torch.no_grad():
                fake = G(melspec_test)

            for i in range(3):
                logger.add_audio(f"test/true_audio{i}", waveforms_test[i], sample_rate=22050)
                logger.add_audio(f"test/pred_audio{i}", fake[i], sample_rate=22050)

            with torch.no_grad():
                fake = G(melspec)

            for i in range(3):
                logger.add_audio(f"true_audio{i}", waveforms[i], sample_rate=22050)
                logger.add_audio(f"pred_audio{i}", fake[i], sample_rate=22050)

            G.train()


if __name__ == '__main__':
    main()
