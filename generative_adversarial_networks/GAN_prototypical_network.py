import sys
sys.path.append('../prototypical_networks_with_autoencoders/')

import os
import math
import pickle
import random
from functools import partial
from typing import Dict, Optional, Callable
from datetime import datetime
# for math operations
import gdown
import numpy as np
from tqdm import tqdm

# ML packages
import torch
import torch.nn as nn
from torch import linalg
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torchvision import datasets, transforms, utils

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

# impelementations
from prototypical_network import ClassificationDataset, PrototypicalNetwork, Convnet
from generative_adversarial import GANSupervisor, Discriminator, Generator

def log(*args, file_path: str=''):
    now = datetime.now()
    # open the file in append mode
    with open(file_path+"log.txt", 'a') as f:
        # write to the file
        f.write(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}\n')
    
    print(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}')

class GANPrototypicalNetwork(PrototypicalNetwork):
    
    def __init__(self, encoder: nn.Module, gan_supervisor: GANSupervisor):
        super(GANPrototypicalNetwork, self).__init__()
        self.gan_supervisor = gan_supervisor
        self.encoder = encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update = False

    def forward(self, support_set, support_classes, query_set, ways, shot, query_num, metric=None, device=None) -> torch.Tensor:
        if metric is None:
            metric = PrototypicalNetwork.pairwise_distances_logits
        if device is None:
            device = self.device
        
        support_embeddings = torch.flatten(self.encoder(support_set), start_dim=1)
        query_embeddings = torch.flatten(self.encoder(query_set), start_dim=1)
        
        # print(f"support_embeddings:{support_embeddings.shape}, query_embeddings:{query_embeddings.shape}")
        centriods = support_embeddings.reshape(ways, shot, -1).mean(dim=1)

        logits = metric(query_embeddings, centriods)
        
        return logits

    def gan_update(self, data):
        X = data.to(self.device)
        # print(X.shape)
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            ])

        D_loss, real_loss, fake_loss = self.gan_supervisor.train_discriminator(transform(X))
        G_loss = self.gan_supervisor.train_generator(transform(X))


    
    def eval_model(self, way, shot, query, num_episode, data_loader, optimizer, metric, criterion: nn.modules.loss, device, mode, output_path, epoch=None):
        n_loss = []
        n_acc = []
        self.gan_supervisor.criterion = nn.BCELoss() 
        with tqdm(range(num_episode), unit="episode") as tepoch:
            if epoch != None: tepoch.set_description(f"Epoch {epoch}")
            for i in tepoch:
                batch = GANPrototypicalNetwork.extract_sample(data_loader= data_loader, ways= way,
                                                           shot= shot, query_num=query, device=device)
                support_set, support_classes, query_set, query_classes = batch

                remap_support_set, remap_support_classes, support_mapping = PrototypicalNetwork.set_consecutive_labels(support_set, support_classes)
                remap_query_set, remap_query_classes, query_mapping = PrototypicalNetwork.set_consecutive_labels(query_set, query_classes)

                # print(f'support_set:{support_set.shape}, support_classes:{support_classes.shape}, query_set:{query_set.shape}, query_classes:{query_set.shape}')
                if mode == "train" or self.update: self.gan_update(support_set)

                logits = self.forward(support_set= remap_support_set, support_classes= remap_support_classes, query_set=remap_query_set,
                                       ways=way, shot= shot, query_num= query, metric=metric, device=device)
                
                
                loss = criterion(logits, remap_query_classes.long())
                acc = GANPrototypicalNetwork.accuracy(logits, remap_query_classes.long())

                n_loss.append(loss.item())
                n_acc.append(acc)
                
                tepoch.set_postfix(mode=mode,  loss=f"({round(np.mean(n_loss), 4):.4f}, {round(np.var(n_loss), 4):.4f})",
                                   acc=f"({np.round(np.mean(n_acc),decimals = 3):.3f}, {np.round(np.var(n_acc),decimals = 3):.3f})")
                
                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    self.hist[f"{mode}_labels"].extend([query_mapping[int(x)] for x in remap_query_classes])
                    self.hist[f"{mode}_logits"].extend([query_mapping[int(x)] for x in np.argmax(logits.cpu().detach().numpy(), axis=1)])
            log(f'{f"[Epoch {epoch}]"if epoch!=None else ""}[{mode}]: acc= ({np.round(np.mean(n_acc),decimals = 3):.3f}, {np.round(1.96*np.std(n_acc)/len(n_acc),decimals = 3):.3f}), loss= ({round(np.mean(n_loss), 4):.4f}, {round(1.96*np.std(n_loss)/len(n_loss), 4):.4f})', file_path=output_path)
            if mode != "train": self.save_best_model(n_acc, n_loss, output_path, epoch)
                    
    def fit(self, train_way: int, train_shot: int, train_query: int, train_num_episode: int,
            test_way: int, test_shot: int, test_query: int, test_num_episode: int, epochs:int, get_dataset: Callable,
            eval_step:int, early_stop:bool, optimizer: torch.optim, lr_scheduler, loss: nn.modules.loss, output_path,) -> None:
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        train_loader = self.data_loader(dataset= get_dataset(mode= 'train'),
                                        way= train_way, shot= train_shot, query= train_query, mode='train')
        val_loader = self.data_loader(dataset= get_dataset(mode= 'val'),
                                      way= test_way, shot= test_shot, query=test_query, mode='val')
        test_loader = self.data_loader(dataset= get_dataset(mode= 'test'),
                                       way= test_way, shot= test_shot, query=test_query, mode='test')
        

        for epoch in range(1, epochs+1,):
            # print("aaa")
            self.train()
            self.__reset_hist__()
            self.eval_model(way= train_way, shot= train_shot, query= train_query, 
                            num_episode=train_num_episode, data_loader= train_loader, optimizer=optimizer,
                            metric=None, criterion= loss, device=device, mode="train", epoch=epoch, output_path=output_path)
            # print(f'train: epoch {epoch}, loss={acc:.4f} acc={loss:.4f}')
            
            if epoch % eval_step == 0:
                
                self.eval()
                self.__reset_hist__()
                self.eval_model(way= test_way, shot= test_shot, query=test_query,
                                num_episode=test_num_episode, data_loader= val_loader, optimizer=None,
                                metric=None, criterion= loss, device=device, mode="val", epoch=epoch, output_path=output_path)
                self.save_classification_report(mode="val", output_path=output_path, epoch=epoch)
                # print(f'eval: epoch {epoch}, loss={acc:.4f} acc={loss:.4f}')
                self.eval()
                self.__reset_hist__()
                self.eval_model(way= test_way, shot= test_shot, query=test_query,
                                num_episode=test_num_episode, data_loader= val_loader, optimizer=None,
                                metric=None, criterion= loss, device=device, mode="test", epoch=epoch, output_path=output_path)
                self.save_classification_report(mode="test", output_path=output_path, epoch=epoch)
            lr_scheduler.step()

        
        self.eval()
        self.__reset_hist__()
        self.eval_model(way= test_way, shot= test_shot, query=test_query,
                        num_episode=test_num_episode, data_loader= test_loader, optimizer=None,
                        metric=None, criterion= loss, device=device, mode="test")
        print(f"best acc: {(self.hist['best_acc'], self.hist['acc_var'])}, loss: {(self.hist['loss'], self.hist['loss_var'])}")
