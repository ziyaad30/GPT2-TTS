import glob
import logging
import os

import torchaudio


def extract_digits(f):
    digits = "".join(filter(str.isdigit, f))
    return int(digits) if digits else -1


def oldest_checkpoint_path(dir_path, regex="ckpt_[0-9]*", preserved=4):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: extract_digits(f))
    if len(f_list) > preserved:
        x = f_list[0]
        return x
    return ""


def latest_checkpoint_path(dir_path, regex="G_[0-9]*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def load_audio(audiopath, sampling_rate):
    audio, sr = torchaudio.load(audiopath)
    # print(f"wave shape: {wave.shape}, sample_rate: {sample_rate}")

    if audio.size(0) > 1:  # mix to mono
        audio = audio[0].unsqueeze(0)

    if sr != sampling_rate:
        try:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        except Exception as e:
            print(f"Warning: {audiopath}, wave shape: {audio.shape}, sample_rate: {sr}")
            return None
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=24000):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


logger = logging