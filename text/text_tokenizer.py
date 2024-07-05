import re
import string

import torch

from text.cleaners import arpa_cleaners

valid_symbols = [
    "[STOP]",
    "[UNK]",
    " ",
    "!",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "?",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "CH",
    "DH",
    "EH",
    "ER",
    "EY",
    "HH",
    "IH",
    "IX",
    "IY",
    "JH",
    "NG",
    "OW",
    "OY",
    "SH",
    "TH",
    "TS",
    "UH",
    "UW",
    "ZH",
]

_valid_symbol_set = set(valid_symbols)
symbols = [s for s in valid_symbols]

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


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


def _symbols_to_sequence(syms):
    return [_symbol_to_id[s] for s in syms]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([s for s in text.split()])


cmu_dict = {}
with open('./text/en_dictionary') as f:
    for entry in f:
        tokens = []
        for t in entry.split():
            tokens.append(t)
        cmu_dict[tokens[0]] = tokens[1:]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([s for s in text.split()])


def sequence_to_text(sequence):
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result += s
    return result


def text_to_sequence(text):
    sequence = []
    text = text.upper()
    text = text.replace('!', ' !')
    text = text.replace('.', ' .')
    text = text.replace(',', ' ,')
    text = text.replace(';', ' ;')
    text = text.replace('?', ' ?')
    text = text.replace(':', ' :')
    text = re.split(r'(\s)', text)

    for phn in text:
        found = False
        for word, pronunciation in cmu_dict.items():
            if word == phn:
                found = True
                for p in pronunciation:
                    try:
                        sequence += _arpabet_to_sequence(p)
                    except:
                        sequence += _arpabet_to_sequence('[UNK]')
                break

        if not found:
            if phn not in string.punctuation:
                if phn == ' ':
                    sequence += _symbols_to_sequence(' ')
                else:
                    # Remove brackets if not word stress
                    if str(phn).__contains__('(') or str(phn).__contains__(')'):
                        open_bracket = phn[:1]
                        close_bracket = phn[-1:]
                        if open_bracket == '(':
                            sequence += _arpabet_to_sequence(open_bracket)
                        phn = phn.replace('(', '')
                        phn = phn.replace(')', '')
                        for word, pronunciation in cmu_dict.items():
                            if word == phn:
                                for p in pronunciation:
                                    sequence += _arpabet_to_sequence(p)
                        if close_bracket == ')':
                            sequence += _arpabet_to_sequence(close_bracket)

                    # this shouldn't happen
                    elif str(phn).__contains__('.'):
                        print(f'stop found in {phn}')
                        with open(f"unknown.txt", 'a', encoding='utf-8') as f:
                            f.write(phn + '\n')
                        phn = phn.replace('.', '')
                        for word, pronunciation in cmu_dict.items():
                            if word == phn:
                                for p in pronunciation:
                                    sequence += _arpabet_to_sequence(p)

                    else:
                        # Unknown words added as [UNK] token and saved to file
                        with open(f"unknown.txt", 'a', encoding='utf-8') as f:
                            f.write(phn + '\n')
                        sequence += _arpabet_to_sequence('[UNK]')
            else:
                # Punctuations added to sequence here
                sequence = sequence[:-1]
                sequence += _symbols_to_sequence(phn)

    return sequence


class TextBpeTokenizer:
    def __init__(self):
        print('Init TextBpeTokenizer')
        self.cleaner = arpa_cleaners
        self.symbols = symbols

    def preprocess_text(self, txt):
        txt = self.cleaner(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        seq = text_to_sequence(txt)
        return seq

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = sequence_to_text(seq)
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')
        return txt

    def vocab_size(self):
        return len(self.symbols)


if __name__ == '__main__':
    tokenizer = TextBpeTokenizer()

    ids = tokenizer.encode("This model sounds really good and above all, it's reasonably fast.")
    print(ids)
    print(tokenizer.decode(ids))

    print("number_text_tokens", tokenizer.vocab_size())

    print(symbols)
