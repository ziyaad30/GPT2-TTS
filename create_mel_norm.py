import torch
import torchaudio
import os

from models.arch_util import TorchCodeMelSpectrogram, TorchMelSpectrogram


# Function to list all .wav files in a directory
def list_wav_files(directory):
    wav_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            wav_files.append(os.path.join(directory, filename))
    return wav_files


torch_mel_spectrogram = TorchMelSpectrogram(
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    sampling_rate=22050,
    mel_fmin=0,
    mel_fmax=8000,
    n_mel_channels=80,
    power=2,
    mel_norm_file=None,
)


# Function to compute and save mel norms
def compute_and_save_mel_norms(directory, output_path):
    # List all .wav files in the directory
    wav_files = list_wav_files(directory)

    # Compute mean of mel spectrograms across all files
    mels = []
    for wav_file in wav_files:
        waveform, sr = torchaudio.load(wav_file)
        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)
        waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
        mel_spec = torch_mel_spectrogram(waveform)
        mels.append(mel_spec.mean((0, 2)).cpu())  # Compute mean across batch and time dimensions

    # Stack and compute overall mean
    mel_norms = torch.stack(mels).mean(0)

    # Save mel norms to .pth file
    torch.save(mel_norms, output_path)
    print(f"Mel norms saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Example directory containing .wav files
    directory_path = "tort_dataset/new_wavs"

    # Output path for saving mel norms
    output_path = "experiments/mel_norms.pth"

    # Compute and save mel norms
    compute_and_save_mel_norms(directory_path, output_path)
