# canvas

A simple, flexible, and modular pytorch template for your deep learning projects. There are multiple templates available for different kinds of machine learning tasks. **Switch to the appropriate branch** from above and see the installation section to download the template:

- Supervised Learning (SL)
- Reinforcement Learning (RL)
- Self-Supervised Learning (SSL)

# Table Of Contents

-  [Installation](#installation)
    - [Requirements](#requirements)
-  [Details](#details)
    -  [Folder Structure](#folder-structure)
    -  [Project Architecture](#project-architecture)
 -  [TODO](#todo)
 -  [Contributing](#contributing)
 -  [License](#license)

# Installation

1.  Clone the repository and go to the project folder.
```
git clone https://github.com/ramanakshay/canvas --depth 1 --branch ssl
cd canvas
```

2. Reset git history.
```
rm -rf .git
git init
git add --all
git commit -m “initial canvas commit”
```

3. Install dependencies from requirements file. Make sure to create a virtual/conda environment before running this command.
```
pip install -r requirements.txt
```

4. Test the code.
```
python main.py
```

## Requirements
- [pytorch](https://pytorch.org/) (An open source deep learning platform)
- [hydra](https://hydra.cc/) (A framework for configuring complex applications)


# Details

## Folder Structure
```
├──  model              - this folder contains all code (networks, layers) of the model
│   ├── weights
│   ├── classifier.py
│   ├── network.py
│   └── layers.py
│
├── data               - this folder contains code relevant to the data and datasets
│   ├── datasets
│   └── data_loader.py
│
├── algorithm             - this folder contains different algorithms of your project
│   ├── train.py
│   └── test.py
│
│
├──  config
│    └── config.yaml  - YAML config file for project
│
│
├──  utils            - this (optional) folder contains utilities of your project
│    └── utils.py
│
│
└── main.py           - entry point of the project

```

## Project Architecture

The main idea behind this template is to model all machine learning tasks as interactions between an agent/model with its environment or external data. All components of the template are built around this core idea. This architecture is inspired from the agent-environment interface in reinforcement learning.

<div align="center">

<img align="center" src="assets/images/architecture.svg">

</div>

# TODO

- [ ] Support for loggers


# Contributing
Any kind of enhancement or contribution is welcomed.


# License

This project is licensed under the MIT License. See LICENSE for more details.
