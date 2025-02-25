import torch
from torch.utils.data import DataLoader
from torchnlp.datasets import multi30k_dataset
from data.preprocessing import load_tokenizers, load_vocab, tokenize, Collator


class TranslateData(object):
    def __init__(self, config):
        self.config = config.data
        self.device = config.settings.device

        spacy_de, spacy_en = load_tokenizers()
        self.tokenizer = {'de': spacy_de, 'en': spacy_en}

        vocab_src, vocab_tgt = load_vocab(self.tokenizer, self.config.vocab_path, self.config.dataset_path)
        self.vocab = {'de': vocab_src, 'en': vocab_tgt}

        collator = Collator(self.tokenizer, self.vocab, device, self.config.max_padding)

        train, val, test = multi30k_dataset(
            directory=config.dataset_path,
            train=True,
            dev=True,
            test=True,
            urls=[
                'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz',
                'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz',
                'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'
            ]
        )

        self.train_dataloader = DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        self.valid_dataloader = DataLoader(
            val,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

    def get_dataloaders(self):
        return {'train': self.train_dataloader, 'valid': self.valid_dataloader}

