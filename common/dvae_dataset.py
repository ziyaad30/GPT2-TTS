import torch
import torch.nn.functional as F
import torchaudio

from models.arch_util import TorchCodeMelSpectrogram, TorchMelSpectrogram


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class DvaeMelDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        print(config['vae_train'])
        self.path = config['vae_train']['train_file']
        self.sample_rate = config['vae_train']['sample_rate']
        self.n_mels = config['vae_train']['n_mels']
        self.pad_to = config['vae_train']['pad_to_samples']
        self.squeeze = config['vae_train']['squeeze']
        self.audiopath_and_text = parse_filelist(self.path)
        self.torch_mel_spectrogram_vocos = TorchCodeMelSpectrogram(n_mel_channels=self.n_mels,
                                                                   sampling_rate=self.sample_rate)

    def __getitem__(self, index):
        audiopath_and_text = self.audiopath_and_text[index]
        wav_file, text = audiopath_and_text[0], audiopath_and_text[1]

        audio, sr = torchaudio.load(wav_file)

        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio).cuda()

        mel = self.torch_mel_spectrogram_vocos(audio.unsqueeze(0))

        if mel.shape[-1] >= self.pad_to:
            start = torch.randint(0, mel.shape[-1] - self.pad_to + 1, (1,))
            mel = mel[:, :, start:start + self.pad_to]
            mask = torch.zeros_like(mel)
        else:
            mask = torch.zeros_like(mel)
            padding_needed = self.pad_to - mel.shape[-1]
            mel = F.pad(mel, (0, padding_needed))
            mask = F.pad(mask, (0, padding_needed), value=1)
        assert mel.shape[-1] == self.pad_to
        if self.squeeze:
            mel = mel.squeeze()

        return mel

    def __len__(self):
        return len(self.audiopath_and_text)
