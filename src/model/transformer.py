import torch
import math
from torch import nn
from torch.nn.functional import log_softmax, pad

def subsequent_mask(size):
    # Mask out subsequent positions. Boolean mask
    attn_shape = (size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 1

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(
                                d_model=config.d_model,
                                nhead=config.nhead,
                                num_encoder_layers=config.num_encoder_layers,
                                num_decoder_layers=config.num_decoder_layers,
                                dim_feedforward=config.dim_feedforward,
                                dropout=config.dropout,
                                batch_first=True) # first dimension is batch_size
        self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab),
                                       PositionalEncoding(config.d_model, config.dropout))
        self.tgt_embed = nn.Sequential(Embeddings(config.d_model, tgt_vocab),
                                       PositionalEncoding(config.d_model, config.dropout))
        self.generator = Generator(config.d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        "Take in and process masked src and target sequences."
        causal_mask = subsequent_mask(tgt.size(-1))
        src, tgt = self.src_embed(src), self.tgt_embed(tgt)
        return self.transformer(src=src, tgt=tgt, tgt_mask=causal_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)

    def encode(self, src, src_mask=None):
        src = self.src_embed(src)
        return self.transformer.encoder(src=src, src_key_padding_mask=src_mask)

    def decode(self, mem, tgt, mem_mask=None, tgt_mask=None):
        causal_mask = subsequent_mask(tgt.size(-1))
        tgt = self.tgt_embed(tgt)
        return self.transformer.decoder(tgt=tgt, memory=mem, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=mem_mask)

