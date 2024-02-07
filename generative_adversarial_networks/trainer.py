import os
import math
import pickle
import random
from functools import partial
from typing import Callable
from collections import OrderedDict

# for math operations
import numpy as np

# ML packages
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# training visualization
import matplotlib.pyplot as plt

# impelementations
from GAN_prototypical_network import GANPrototypicalNetwork
from generative_adversarial import GANSupervisor, Discriminator, Generator

state = 42
torch.manual_seed(state)
torch.cuda.manual_seed(state)
np.random.seed(state)
random.seed(state)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def log(*args, file_path: str=''):
    now = datetime.now()
    # open the file in append mode
    with open(file_path+"log.txt", 'a') as f:
        # write to the file
        f.write(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}\n')
    
    print(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}')

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_bird_dataset(path: str= "../../datasets/birds/",
                        mode: str= "train") -> torch.utils.data.Dataset:
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((96,96)),
         # transforms.Normalize((0.5), (0.5))
        ])

    # transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(path+f'/{mode}/',
                                        transform=transform)
    
    return dataset
    
def get_mini_imagenet_dataset(path: str= "../../datasets/mini_imagenet/",
                        mode: str= "train") -> torch.utils.data.Dataset:
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((96,96), antialias=True),
         transforms.Normalize((0.5), (0.5))
        ])

    # transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(path+f'/{mode}/',
                                        transform=transform)
    dataset.y = [64] if mode == "train" else [20] if mode == "test" else [16]
    return dataset

def get_tiered_imagenet_dataset(path: str= "../../datasets/tiered_imagenet/",
                        mode: str= "train") -> torch.utils.data.Dataset:
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((96,96), antialias=True),
         transforms.Normalize((0.5), (0.5))
        ])

    # transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(path+f'/{mode}/',
                                        transform=transform)
    dataset.y = [351] if mode == "train" else [160] if mode == "test" else [97]
    return dataset

def get_fc100_dataset(path: str= "../../datasets/FC100/",
                        mode: str= "train") -> torch.utils.data.Dataset:
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((96,96), antialias=True),
         transforms.Normalize((0.5), (0.5))
        ])

    # transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(path+f'/{mode}/',
                                        transform=transform)
    dataset.y = [60] if mode == "train" else [20] if mode == "test" else [20]
    return dataset

def check_folder_loader(dataset: torch.utils.data.Dataset, grid_split: int= 10, num_img: int= 30) -> None:
    label_mapping = {y: x for x, y in dataset.class_to_idx.items()}

    fraction = math.ceil(num_img/grid_split)
    width = grid_split if fraction != 0 else num_img
    height = fraction if  width == grid_split else 1

    figure = plt.figure(figsize=(width, height))
    for i, (img, label) in enumerate(dataset):
        if i==0: print(f"image.shape: {img.shape}")
        elif i>=num_img: break
        
        figure.add_subplot(height, width, i+1)
        plt.title(label_mapping[label], fontsize=4)
        plt.axis("off")
        plt.imshow(img[0].numpy(), cmap="gray")
        

def check_MetaDataset_loader(dataset: torch.utils.data.Dataset, grid_split: int= 10, num_img: int= 30) -> None:
    sample = dataset.sample()
    fraction = math.ceil(num_img/grid_split)
    width = grid_split if fraction != 0 else num_img
    height = fraction if  width == grid_split else 1

    figure = plt.figure(figsize=(width, height))
    for i, img in enumerate(sample[0]):
        if i==0: print(f"image.shape: {img.shape}")
        elif i>=num_img: break
        
        figure.add_subplot(height, width, i+1)
        plt.title(sample[1][i].numpy(), fontsize=4)
        plt.axis("off")
        plt.imshow(img[0].numpy(), cmap="gray")

def GAN_train(num_ways: int, num_shot: int, get_dataset: Callable, experiment_name: str,
          autoencoder_path: str="../outputs/exported/resnet_autoencoder/autoencoder_withouthist.pt"):
    print(f"\n ================== [PrototypicalAutoencoderNetwork][{experiment_name}] {num_ways}-WAY {num_shot}-SHOT ================== \n")
    torch.cuda.empty_cache()

    autoencoder = ResNetAutoEncoder(arch= "resnet18", num_workers=-1,
                                batch_size=1024, eval_step=1)
    
    if autoencoder_path == "resnet18":
        
        checkpoint = torch.load("../outputs/exported/resnet_autoencoder/caltech256-resnet18.pth")

        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        autoencoder.load_state_dict(new_state_dict)

    else:
        autoencoder = torch.load(autoencoder_path)

    autoencoder.eval()

    gen =  Generator(input_size=100, feature_size=96, num_channels=3)
    dis =  Discriminator(feature_size=96, num_channels=3,)

    supervisor = GANSupervisor(generator= gen,
                discriminator= dis,
                generator_optimizer= torch.optim.Adam,
                discriminator_optimizer= torch.optim.Adam,
                generator_learning_rate= 1e-8,
                discriminator_learning_rate=1e-8,
                get_dataset= get_mini_imagenet_dataset,
                get_noise_generator=None,
                epoch= 100,
                batch_size= 64,
                embedding_size= 100,
                )

    model = GANPrototypicalNetwork(encoder= autoencoder.encoder, gan_supervisor=supervisor)
    output_dir = "../outputs/exported/"
    save_dir = f"{output_dir}/{model.__class__.__name__}/{experiment_name}-{num_ways}way{num_shot}shot15query/"
    create_folder(f"{output_dir}/{model.__class__.__name__}/")
    create_folder(save_dir)

    # best parameters lr=1e-4, epochs= 5, eval_step= 1, episode=10000, step_size=10000, autoencoder_lr=1e-6, autoencoder_weight_decay= 1e-8,
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #divide the learning rate by 2 at each epoch, as suggested in paper
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
    loss = F.cross_entropy

    model.fit(get_dataset=get_dataset, epochs= 5, train_num_episode=10000, test_num_episode= 10000,
                train_way= num_ways, train_shot= num_shot, train_query= 15, test_way= num_ways, test_shot= num_shot, test_query= 15,
                eval_step= 1, early_stop= True, optimizer= optimizer, lr_scheduler=lr_scheduler, loss= loss,
                output_path=save_dir,)


if __name__ == "__main__":

################################## MINI-IMAGENET ##################################
    experiment_name = 'mini_imagenet'
    get_dataset = partial(get_mini_imagenet_dataset, path=f"../../datasets/{experiment_name}/",)
    
    GAN_train(num_ways= 5, num_shot= 5,
                            get_dataset= get_dataset, 
                            experiment_name= experiment_name,
                            autoencoder_path="../outputs/exported/resnet_autoencoder/autoencoder_withouthist.pt"
                            )

    GAN_train(num_ways= 5, num_shot= 1,
                            get_dataset= get_dataset, 
                            experiment_name= experiment_name,
                            autoencoder_path="../outputs/exported/resnet_autoencoder/autoencoder_withouthist.pt")

###################################### FC100 ######################################
    experiment_name = 'FC100'
    get_dataset = partial(get_fc100_dataset, path=f"../../datasets/{experiment_name}/",)

    GAN_train(num_ways= 5, num_shot= 5,
                            get_dataset= get_dataset, 
                            experiment_name= experiment_name,
                            autoencoder_path="../outputs/exported/resnet_autoencoder/autoencoder_withouthist.pt")

    GAN_train(num_ways= 5, num_shot= 1,
                            get_dataset= get_dataset, 
                            experiment_name= experiment_name,
                            autoencoder_path="../outputs/exported/resnet_autoencoder/autoencoder_withouthist.pt")

    
################################# TIERED IMAGENET #################################
    experiment_name = 'tiered_imagenet'
    get_dataset = partial(get_tiered_imagenet_dataset, path=f"../../datasets/{experiment_name}/",)

    GAN_train(num_ways= 5, num_shot= 5,
                            get_dataset= get_dataset, 
                            experiment_name= experiment_name,
                            autoencoder_path="../outputs/exported/resnet_autoencoder/autoencoder_withouthist.pt")
    
    GAN_train(num_ways= 5, num_shot= 1,
                            get_dataset= get_dataset, 
                            experiment_name= experiment_name,
                            autoencoder_path="../outputs/exported/resnet_autoencoder/autoencoder_withouthist.pt")