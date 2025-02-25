import torch
from torch import nn

from model.transformer import Transformer


class Translator(object):
    def __init__(self, src_vocab, tgt_vocab, config):
        self.config = config.model
        self.transformer = Transformer(src_vocab, tgt_vocab,
                                       d_model=self.config.d_model,
                                       nhead=self.config.nhead,
                                       num_encoder_layers=self.config.num_encoder_layers,
                                       num_decoder_layers=self.config.num_decoder_layers,
                                       dim_feedforward=self.config.dim_feedforward,
                                       dropout=self.config.dropout)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

    def predict(self, src, tgt, src_mask, tgt_mask):
        return self.transformer(src, tgt, src_mask, tgt_mask)

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        memory = self.transformer.encode(src, src_mask)
        ys = torch.zeros(1,1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = self.transformer.decode(memory, ys, src_mask)
            prob = self.transformer.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.zeros(1,1).type_as(src.data).fill_(next_word)], dim=1)
        return ys