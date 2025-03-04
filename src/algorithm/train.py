import time
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from algorithm.utils import Batch
from algorithm.loss import SimpleLossCompute, LabelSmoothing

import inspect

def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

def estimate_mfu(self, fwdbwd_per_iter, dt):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu

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

        criterion = LabelSmoothing(len(self.data.vocab['en']), padding_idx=0, smoothing=self.config.smoothing)
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
        self.accum_iter = self.config.accum_iter
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
        return total_loss / total_tokens

    def run(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            self.run_epoch(mode='train')
            self.model.eval()
            self.run_epoch(mode='valid')
