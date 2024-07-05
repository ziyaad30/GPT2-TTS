import re
from unidecode import unidecode

from text.numbers_ import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misses"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("sr", "senior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ('inc', 'incorporated'),
        ('etc', 'etcetera'),
    ]
]


def expand_abbreviations(text):
    """Check and replace common abbreviations with their full words."""
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = text.strip()
    text = lowercase(text)
    text = convert_to_ascii(text)
    text = text.replace(':', ' ')
    text = text.replace('"', '')
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = text.strip()
    text = convert_to_ascii(text)
    text = clean_apos(text)
    text = text.replace('-', ' ')
    text = text.replace('"', '')
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def check_arpa_stress(text):
    new_text = []
    for test in text.split(" "):
        if test.__contains__('(2)'):
            new_text.append(test)
        else:
            new_text.append(expand_numbers(test))
    return_text = ' '.join(new_text)
    return return_text


def clean_apos(text):
    rgx = re.compile(r"(?<!\w)\'|\'(?!\w)")
    text = rgx.sub('', text)
    return text


def arpa_cleaners(text):
    """Pipeline for English text, including abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = clean_apos(text)
    text = expand_abbreviations(text)
    text = text.replace('-', ' ')
    text = text.replace('"', '')
    text = text.replace(':', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('’', "'")
    text = check_arpa_stress(text)
    text = collapse_whitespace(text)
    return text


def replace_symbols(text):
    # text = text.replace('"', '')
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", " ")
    text = text.replace("&", " and ")
    return text


def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text
