import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import inspect

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
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

def estimate_mfu(model, fwdbwd_per_iter, dt):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = model.get_num_params()
    cfg = model.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu

def get_lr(it, warmup_iters, lr_decay_iters, min_lr, lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)

class Trainer:
    def __init__(self, data, model, config):
        self.data = data
        self.model = model
        self.config = config.algorithm
        self.device = config.system.device

        self.loss_func = (lambda logits, targets:
                         F.cross_entropy(
                             logits.view(-1, logits.size(-1)),
                             targets.view(-1)))

        self.optimizer = torch.optim.AdamW(self.model.gpt.parameters(),
                                           lr=self.config.lr,
                                           betas=(0.9, 0.95),eps=1e-8)

        self.scheduler = LambdaLR(optimizer=self.optimizer,
                                  lr_lambda=lambda it: rate(
                                      it, self.config.warmup_iters, self.config.lr_decay_iters,
                                      self.config.min_lr, self.config.lr
                                  ))

    def run_epoch(self):

        for i in range(50):
            ts = time.time()
            x,y = self.data.get_batch()
            x,y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, targets = self.model.predict(x), y
                loss = self.loss_func(logits, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.gpt.parameters(), 1.0)
            self.optimizer.step()
            if self.config.decay_lr:
                self.scheduler.step()
            torch.cuda.synchronize()
            te = time.time()
            dt = (te-ts)*1000
            tokens_per_sec = (self.data.batch_size * self.data.block_size) / (te - ts)
            print(f"Step {i}, Loss: {loss.item()}| norm: {norm:.4f}| dt: {dt:.2f}ms, tok/sec: {tokens_per_sec}")
        

