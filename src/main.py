from algorithm.train import Trainer
from model.translator import Translator
from data.data import TranslateData

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATA ##
    data = TranslateData(config.data)
    print('Data Loaded.')

    # ## MODEL ##
    src_vocab, tgt_vocab = len(data.vocab['de']), len(data.vocab['en'])
    model = Translator(src_vocab, tgt_vocab, config.model)
    print('Model Created.')
    #
    # ## ALGORITHM ##
    algorithm = Trainer(data, model, config.algorithm)
    algorithm.run()
    print('Done!')

if __name__ == "__main__":
    main()
