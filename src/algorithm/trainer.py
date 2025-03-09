import time
import torch
import torch.nn.functional as F

from algorithm.optimizer import configure_optimizers, get_lr

class Trainer:
    def __init__(self, data, model, config, state, ctx):
        self.data = data
        self.model = model
        self.config = config.algorithm
        self.state = state
        self.ctx = ctx

        self.optimizer = configure_optimizers(
                    self.model.gpt,
                    weight_decay=self.config.weight_decay,
                    learning_rate=self.config.lr,
                    betas=tuple(self.config.betas),
                    eps=self.config.eps,
                    state=self.state)

        self.loss_func = (lambda logits, targets:
                              F.cross_entropy(
                                  logits.view(-1, logits.size(-1)),
                                  targets.view(-1)))

        self.scaler = torch.amp.GradScaler(self.state.device_type,
                                           init_scale=65536.0,
                                           growth_factor=2.0,
                                           backoff_factor=0.5,
                                           growth_interval=2000,
                                           enabled=(config.system.dtype == 'float16'))

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.data.get_batch(split)
                with self.ctx:
                    logits, targets = self.model.predict(x), y
                    loss = self.loss_func(logits, targets)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def run_batch(self):
        self.optimizer.zero_grad()
        for micro_step in range(self.config.grad_accum):
            if self.state.is_ddp:
                self.model.gpt.require_backward_grad_sync = (micro_step == self.config.grad_accum - 1)
            x,y = self.data.get_batch()
            with self.ctx:
                logits, targets = self.model.predict(x), y
                loss = self.loss_func(logits, targets)
                loss = loss / self.config.grad_accum
            self.scaler.scale(loss).backward()
        # grad clip
        grad_clip = self.config.grad_clip
        if grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.gpt.parameters(), grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def run(self):
        for step in range(self.config.max_iters):
            lr = get_lr(step, self.config.warmup_iters,
                        self.config.lr_decay_iters,
                        self.config.min_lr,
                        self.config.lr) if self.config.decay_lr else self.config.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            ts = time.time()
            loss = self.run_batch()
            torch.cuda.synchronize()
            te = time.time()

            if self.state.is_master_process:
                dt = (te-ts)*1000
                loss_accum = loss.item() * self.config.grad_accum # approximate loss
                tokens_per_sec = (self.data.batch_size * self.data.block_size * self.config.grad_accum * self.state.world_size) / (te - ts)
                if step % self.config.eval_interval == 0:
                    losses = self.estimate_loss()
                    print(f"val step {step}, train loss: {losses['train']:.4f} | val loss {losses['val']:.4f}")
                print(f"step {step}, loss: {loss_accum:.4f} | time: {dt:.2f}ms | toks/sec: {tokens_per_sec:0.0f}")
        

