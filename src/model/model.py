import torch
import torch.nn.functional as F

from model.nanogpt import GPT


class GPTModel:
    def __init__(self, config):
        self.config = config.model
        self.gpt = GPT(self.config)

        if self.config.from_pretrained:
            self.gpt.load_pretrained(self.config.model_name)

        if config.data.block_size < config.model.block_size:
            self.gpt.crop_block_size(config.data.block_size)

        print("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def train(self):
        self.gpt.train()

    def eval(self):
        self.gpt.eval()
        
    def to(self, device):
        self.gpt.to(device)

    def get_num_params(self, non_embedding=True):
        return self.gpt.get_num_params(non_embedding)

    def predict(self, idx):
        return self.gpt(idx, targets=True)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self.gpt(idx_cond, targets=False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
