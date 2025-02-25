class Batch(object):
    """
        Object for holding a batch of data with mask during training.
    """
    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src == pad)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = (self.tgt == pad)
            self.ntokens = (self.tgt_y != pad).data.sum()