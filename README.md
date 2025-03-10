# nanoGPT: Generative Pre-trained Transformer

Generative Pre-trained Transformers (GPTs) are a class of next-word prediction models based on the transformer architecture. This projects reimplements [nanoGPT](https://github.com/karpathy/nanoGPT), a repository to train/finetune medium-sized GPT models.

<p align="center">
  <img width="40%" alt="GPT Architecture" src="assets/gpt_architecture.svg" >
</p>


## Data

The GPT-2 model is trained on the OpenWebText dataset, an open reproduction of OpenAI's (private) WebText. You can also finetune a pretrained GPT-2 model on the TinyShakespeare dataset, which consists of the works of Shakespeare. The dataset is tokenized using the BPE tokenizer from the tiktoken library. Run the command below to download and tokenize the dataset:

```
# navigate to the datasets folder
cd src/data/datasets

# download openwebtext dataset
python openwebtext/prepare.py

# download shakespeare dataset
python shakespeare/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Once downloaded, update `data_dir` in the config with the absolute path of the folder containing the bin files.

```
# inside train_gpt2.yaml
...
data:
    data_dir: path_to_bin_files
...
```

## Model

The GPT model supports the following methods:

**`model.predict(idx: Tensor, targets=True: Boolean)`**

Takes a conditioning sequence of indices idx (LongTensor of shape (b,t)), returns target prediction for each input token (`targets=True`) or just for the last token (`targets=False`)

**`model.generate(idx: LongTensor, max_new_tokens: Integer, temperature=1.0: Integer, top_k=None: Integer)`**

Takes a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete the sequence max_new_tokens times, feeding the predictions back into the model each time.

You can initialize the model from pretrained HuggingFace checkpoints. Select any of pretrained models listed in the `config/model` folder by updating the config as follows:

```
├── config/model             
│   ├── gpt2-large.yaml
│   ├── gpt2-medium.yaml
│   ├── gpt2-xl.yaml
│   └── gpt2.yaml


# inside finetune_shakespeare.yaml
defaults:
    - model: gpt2-xl
    - _self_
...
```


## To-dos

Any kind of enhancement or contribution is welcomed.

- [ ] Evaluation script
- [ ] Benchmarking script
- [ ] Support for loggers
      
## References




