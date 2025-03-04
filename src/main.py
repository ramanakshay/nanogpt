from model import GPTModel
import tiktoken
import torch


import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    # ## DATA ##
    # data = TranslateData(config)
    # print('Data Loaded.')

    num_return_sequences = 5
    max_length = 30

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I am a large language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to('cuda')


    # ## MODEL ##
    model = GPTModel(config)
    model.eval()
    model.to('cuda')
    print('Model Created.')

    # ## ALGORITHM ##
    # algorithm = Trainer(data, model, config)
    # algorithm.run()
    # print('Done!')


if __name__ == "__main__":
    main()
