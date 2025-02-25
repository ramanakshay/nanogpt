import time
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from algorithm.utils import Batch
from algorithm.loss import SimpleLossCompute, LabelSmoothing

def rate(step, model_size, factor, warmup):
    """
we have to default the step to 1 for LambdaLR function
to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class Trainer(object):
    def __init__(self, data, model, config):
        self.data = data
        self.model = model
        self.config = config.algorithm
        self.dataloaders = data.get_dataloaders()

        criterion = LabelSmoothing(len(self.data.vocab['en']) , padding_idx=0, smoothing=self.config.smoothing)
        self.loss = SimpleLossCompute(self.model.transformer.generator, criterion)
        self.optimizer = torch.optim.Adam(
            self.model.transformer.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-9
        )
        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: rate(
                step, model_size=self.model.transformer.src_embed[0].d_model, factor=1.0, warmup=self.config.warmup
            ),
        )
        self.accum_iter = config.accum_iter
        self.train_state = TrainState()

    def run_epoch(self, mode):
        self.model.train()
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        train_state = self.train_state


        for i, b in enumerate(self.dataloaders[mode]):
            batch = Batch(b[0], b[1], pad=2)
            out = self.model.predict(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = self.loss(out, batch.tgt_y, batch.ntokens)

            if mode == 'train':
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens
                if i % self.accum_iter == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                self.scheduler.step()

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 40 == 1 and (mode == "train"):
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                start = time.time()
                tokens = 0

            del loss
            del loss_node
        return total_loss/total_tokens


    def run(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            self.run_epoch(mode='train')
            self.model.eval()
            self.run_epoch(mode='valid')