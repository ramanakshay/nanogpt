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
    device = config.system.device

    num_return_sequences = 5
    max_length = 30

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode("Hello, I am a large language model,")
    context_size = len(tokens)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)


    # ## MODEL ##
    model = GPTModel(config)
    model.eval()
    model.to(device)
    print('Model Created.')

    x = model.generate(x, max_length, 1.0, 50)


    for i in range(num_return_sequences):
        tokens = x[i, :(context_size+max_length)].tolist()
        decoded = tokenizer.decode(tokens)
        print(">", decoded)

    # ## ALGORITHM ##
    # algorithm = Trainer(data, model, config)
    # algorithm.run()
    # print('Done!')


if __name__ == "__main__":
    main()
