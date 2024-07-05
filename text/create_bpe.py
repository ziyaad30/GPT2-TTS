from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase

bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer
bpe_trainer = BpeTrainer(special_tokens=["[STOP]", "[UNK]", "[SPACE]"], vocab_size=255)

from tokenizers.pre_tokenizers import Whitespace
bpe_tokenizer.pre_tokenizer = Whitespace()
bpe_tokenizer.normalizer = normalizers.Sequence([Lowercase()])

bpe_tokenizer.train(["C:\\Users\\User\\PycharmProjects\\pdfBot\\test_dataset\\test.txt"], bpe_trainer)
bpe_tokenizer.save("./test.json")
