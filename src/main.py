from data import TextData
from model import GPTModel
from algorithm import Trainer

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    # ## DATA ##
    data = TextData(config)
    print('Data Loaded.')

    # ## MODEL ##
    model = GPTModel(config)
    print('Model Created.')

    # ## ALGORITHM ##
    algorithm = Trainer(data, model, config)
    algorithm.run_epoch()
    print('Done!')


if __name__ == "__main__":
    main()
