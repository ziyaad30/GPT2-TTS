import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils import load_audio


def get_prompt_slice(gt_path, min_sample_length, max_sample_length, sample_rate=22050):
    rel_clip = load_audio(gt_path, sample_rate)
    sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length

    if gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    rel_clip = F.pad(rel_clip, pad=(0, max_sample_length - rel_clip.shape[-1]))

    return rel_clip, rel_clip.shape[-1]


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class XTTSDataset(Dataset):
    def __init__(self, config, tokenizer, is_eval=False):
        self.is_eval = is_eval
        if not self.is_eval:
            self.path = config['gpt_train']['train_file']
        else:
            self.path = config['gpt_train']['valid_file']
        self.audiopath_and_text = parse_filelist(self.path)
        self.tokenizer = tokenizer
        self.sample_rate = config['vae_train']['sample_rate']
        self.max_wav_len = 255995
        self.max_text_len = 200
        self.min_conditioning_length = 66150
        self.max_conditioning_length = 132300

    def get_text(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        assert not torch.any(tokens == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tokens)}"
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0), f"Stop token found in {text}"
        return tokens

    def __getitem__(self, index):
        audiopath_and_text = self.audiopath_and_text[index]
        audio_path, text = audiopath_and_text[0], audiopath_and_text[1]

        wav = load_audio(audio_path, self.sample_rate)

        tseq = self.get_text(text)

        # Basically, this audio file is nonexistent or too long to be supported by the dataset.
        if wav is None:
            print(f"{audio_path} does not exist")
            return

        if self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len:
            print(f"Wave Length: {wav.shape[1]} > {self.max_wav_len} for {audio_path}")
            return

        if self.max_text_len is not None and tseq.shape[0] > self.max_text_len:
            print(f"Sequence Length: {tseq.shape[0]} > {self.max_text_len} for {audio_path}")
            return

        cond, cond_len = get_prompt_slice(audio_path,
                                          min_sample_length=self.min_conditioning_length,
                                          max_sample_length=self.max_conditioning_length,
                                          sample_rate=self.sample_rate)

        res = {
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audio_path,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long),
        }
        return res

    def __len__(self):
        return len(self.audiopath_and_text)

    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)
        try:
            batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        except Exception as e:
            print(e)

        # stack for features that already have the same shape
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        batch["text_lengths"] = torch.stack(batch["text_lengths"])
        batch["conditioning"] = torch.stack(batch["conditioning"])
        batch["cond_lens"] = torch.stack(batch["cond_lens"])

        if torch.any(batch["cond_lens"].isnan()):
            batch["cond_lens"] = None

        max_text_len = batch["text_lengths"].max()
        max_wav_len = batch["wav_lengths"].max()

        # create padding tensors
        text_padded = torch.IntTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        text_padded = text_padded.zero_()
        wav_padded = wav_padded.zero_()
        for i in range(B):
            text = batch["text"][i]
            text_padded[i, : batch["text_lengths"][i]] = torch.IntTensor(text)
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)

        batch["wav"] = wav_padded
        batch["padded_text"] = text_padded
        return batch
