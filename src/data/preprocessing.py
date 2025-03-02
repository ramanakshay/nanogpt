import os
import spacy
import torch
from os.path import exists
from torch.nn.functional import pad
from torchtext.vocab import build_vocab_from_iterator
from torchnlp.datasets import multi30k_dataset


def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en, dataset_path):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    train, val, test = multi30k_dataset(
        directory=dataset_path,
        train=True,
        dev=True,
        test=True,
        urls=[
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'
        ]
    )

    print("Building German Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index='de'),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index='en'),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(tokenizers, vocab_path, dataset_path):
    spacy_de, spacy_en = tokenizers['de'], tokenizers['en']
    if not exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en, dataset_path)
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        vocab_src, vocab_tgt = torch.load(vocab_path)
    print("Vocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


class Collator(object):
    def __init__(self, tokenizer, vocab, device, max_padding):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        self.max_padding = max_padding

    def __call__(self, batch):
        bs_id = torch.tensor([0], device=self.device)  # <s> token id
        eos_id = torch.tensor([1], device=self.device)  # </s> token id
        pad_id = 2
        src_list, tgt_list = [], []
        for s in batch:
            _src, _tgt = s['de'], s['en']
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab['de'](tokenize(_src, self.tokenizer['de'])),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab['en'](tokenize(_tgt, self.tokenizer['en'])),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(
                    processed_src,
                    (
                        0,
                        self.max_padding - len(processed_src),
                    ),
                    value=pad_id,
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, self.max_padding - len(processed_tgt)),
                    value=pad_id,
                )
            )

        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        return src, tgt
