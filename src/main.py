from data import TextData
from model import GPTModel
from algorithm import Trainer

import torch

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    print(config)
    # ## DATA ##
    # data = TextData(config)
    # print('Data loaded.')
    #
    # ## MODEL ##
    # model = GPTModel(config)
    #
    # torch.set_float32_matmul_precision('high')
    #
    # print('Model created.')
    #
    # ## ALGORITHM ##
    # algorithm = Trainer(data, model, config)
    # algorithm.run_epoch()
    # print('Done!')


if __name__ == "__main__":
    main()
