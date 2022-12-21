from torch.utils.data import DataLoader
import torch
from argparse import ArgumentParser

from datasets.test_dataset import TestDataset
from datasets.ljspeech_dataset import collate_fn
from configs.train_config import Config
from utils.utils import set_random_seed
from melspec.melspec import MelSpectrogram, MelSpectrogramConfig
from model.generator import Generator
from utils.wandb_writer import WanDBWriter


def main(G_path):
    train_config = Config()
    melspec_config = MelSpectrogramConfig()
    set_random_seed(3407)
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
    G = Generator(train_config)
    G.load_state_dict(torch.load(G_path)['Generator'])
    G.to(train_config.device)
    G.eval()
    with torch.no_grad():
        fake = G(melspec_test)

    logger = WanDBWriter(train_config)
    for i in range(3):
        logger.add_audio(f"test/true_audio{i}", waveforms_test[i], sample_rate=22050)
        logger.add_audio(f"test/pred_audio{i}", fake[i], sample_rate=22050)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--G_path", type=str, help="path to model weights")
    args = parser.parse_args()
    main(args.G_path)
