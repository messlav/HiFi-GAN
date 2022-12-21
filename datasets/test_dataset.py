from torch.utils.data import Dataset
import torchaudio
import torch
import os
from torch.utils.data import DataLoader
from melspec.melspec import MelSpectrogram, MelSpectrogramConfig
from datasets.ljspeech_dataset import collate_fn


class TestDataset(Dataset):
    def __init__(self, dir_name):
        super(TestDataset, self).__init__()
        self.dir_name = dir_name
        self.list_files = os.listdir(dir_name)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        file = self.list_files[index]
        file_path = os.path.join(self.dir_name, file)

        waveform, rate_of_sample = torchaudio.load(file_path)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        return waveform, waveform_length


def test():
    dataset = TestDataset('../val/wavs')
    waveform, waveform_length = next(iter(dataset))
    print(waveform.shape, waveform_length)
    print('done dataset')
    data_loader = DataLoader(
        dataset,
        batch_size=3,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
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
