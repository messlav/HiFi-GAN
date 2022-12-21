import torch
from torchaudio.datasets import LJSPEECH
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from melspec.melspec import MelSpectrogram, MelSpectrogramConfig


class LJSpeechDataset(LJSPEECH):
    def __init__(self, root, max_len: int = 8192 * 2):
        super().__init__(root=root)
        self.max_len = max_len

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        if self.max_len is not None:
            start_idx = random.randint(0, waveform.shape[1] - self.max_len)
            waveform = waveform[:, start_idx: start_idx + self.max_len]

        return waveform, waveform_length


def collate_fn(batch):
    waveform, waveform_length = list(
        zip(*batch)
    )

    waveform = pad_sequence([
        waveform_[0] for waveform_ in waveform
    ]).transpose(0, 1)
    waveform_length = torch.cat(waveform_length)

    return waveform, waveform_length


def test():
    dataset = LJSpeechDataset('../')
    waveform, waveform_length = next(iter(dataset))
    print(waveform.shape, waveform_length)
    print('done dataset')
    data_loader = DataLoader(
        dataset,
        batch_size=10,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    waveforms, waveforms_length = next(iter(data_loader))
    print(waveforms.shape, waveforms_length.shape)
    print('done dataloader')

    melspectrogram = MelSpectrogram(MelSpectrogramConfig())
    melspec = melspectrogram(waveforms)
    print(melspec.shape)
    print('done melspec')


if __name__ == '__main__':
    test()
