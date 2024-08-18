# Meta-Tasks: An alternative view on Meta-Learning Regularization
--------


--------

## Brief Introduction

pass

## Installation

Use `pip install -r requirements.txt` or for conda `conda install --file conda_env.txt`

```
conda create -n few python=3.8
conda activate few
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
```

## Description

Experiments results can be seen in 'results_comparing.xslx' and 'results_comparing_2.xslx'

Most implementations are in the folder `prototypical_networks_with_autoencoders`. The `py` files are the implementations and `ipynb files are for testing. 

--------

autoencoder -> backbone of autoencoders taken and modified from [Horizon2333](https://github.com/Horizon2333/imagenet-autoencoder/blob/main/models/resnet.py)
If there was any pre-trained model that wasn't pushed to the repo you can use [Horizon2333](https://github.com/Horizon2333/imagenet-autoencoder/blob/main/models/resnet.py) repo

prototypical network -> meta-learning implementation of a prototypical network by learn2learn package but heavily edited since it had a lot of bugs and limitations.

autoencoder_prtotypical -> This is the implementation of our idea.

base -> Was a backup code framework that I changed after week 2 and just saved as a back, its primary purpose is a simple prototypical network.

## Testing
For General use, you must first train an autoencoder as a backbone and then train the few-shot classifier (Prototypical Classifier).

For recreating the results from the paper you must first train an autoencoder in `prototypical_networks_with_autoencoders/autoencoder.ipynb` and then use it to train the model in `prototypical_networks_with_autoencoders/autoencoder_prototypical_network.ipynb`.

To compare this paper's results with the normal Prototypical Network, you can train your encoder with `prototypical_networks_with_autoencoders/prototypical_network.ipynb`


