import re

import torch
from tokenizers import Tokenizer

from text.cleaners import english_cleaners

DEFAULT_VOCAB_FILE = 'data/tokenizer.json'


def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(',
        '}': ')',
        '[': '(',
        ']': ')',
        '—': '-',
        '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]),
                         flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None):
        if vocab_file is not None:
            print(f">>> Vocab file {vocab_file} loaded")
        self.tokenizer = Tokenizer.from_file(
            DEFAULT_VOCAB_FILE if vocab_file is None else vocab_file
        )

    def preprocess_text(self, txt):
        txt = english_cleaners(txt)
        txt = remove_extraneous_punctuation(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        txt = txt.replace(' ', '[SPACE]')
        ids = self.tokenizer.encode(txt).ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        # txt = txt.replace('[UNK]', '')
        return txt

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values())


if __name__ == '__main__':
    tokenizer = VoiceBpeTokenizer()
    print(tokenizer.tokenizer.get_vocab_size())
