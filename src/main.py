from model import GPTModel

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    # ## DATA ##
    # data = TranslateData(config)
    # print('Data Loaded.')

    # ## MODEL ##
    model = GPTModel(config)
    print('Model Created.')

    # ## ALGORITHM ##
    # algorithm = Trainer(data, model, config)
    # algorithm.run()
    # print('Done!')


if __name__ == "__main__":
    main()
