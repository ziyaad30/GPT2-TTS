import torch
import torch.nn.functional as F
import torchaudio

from models.arch_util import TorchCodeMelSpectrogram, TorchMelSpectrogram
from vocoder2.feature_extractors import MelSpectrogramFeatures


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


torch_mel_spectrogram_dvae = TorchMelSpectrogram(
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    sampling_rate=22050,
    mel_fmin=0,
    mel_fmax=8000,
    n_mel_channels=80,
    power=2,
    mel_norm_file="experiments/mel_norms.pth",
)


class DvaeMelDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        print(config['vae_train'])
        self.path = config['vae_train']['train_file']
        self.sample_rate = 22050  # config['vae_train']['sample_rate']
        self.pad_to = config['vae_train']['pad_to_samples']
        self.squeeze = config['vae_train']['squeeze']
        self.audiopath_and_text = parse_filelist(self.path)
        # self.torch_mel_spectrogram_vocos = TorchCodeMelSpectrogram(n_mel_channels=self.n_mels,
        #                                                            sampling_rate=self.sample_rate)
        self.mel_extractor = MelSpectrogramFeatures().cuda()

    def __getitem__(self, index):
        audiopath_and_text = self.audiopath_and_text[index]
        wav_file, text = audiopath_and_text[0], audiopath_and_text[1]

        audio, sr = torchaudio.load(wav_file)

        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio).cuda()

        mel = torch_mel_spectrogram_dvae(audio)

        # if self.squeeze:
        #     mel = self.mel_extractor(audio)
        # else:
        #     mel = self.mel_extractor(audio.unsqueeze(0))

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
