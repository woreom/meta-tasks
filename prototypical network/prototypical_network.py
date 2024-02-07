#!/usr/bin/env python
# coding: utf-8
# from __future__ import annotations

import os
import argparse
import pickle
from functools import partial
from dataclasses import dataclass, field
from abc import ABC, abstractclassmethod
from typing import Dict, List, Tuple, Optional, Callable, Iterable
# from collections.abc import Callable, Iterable, Generator

import gdown
import statistics
from matplotlib.pyplot import flag
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from PIL import Image
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import classification_report


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from learn2learn.vision.transforms import RandomClassRotation


def log(*args, file_path: str=''):
    now = datetime.now()
    # open the file in append mode
    with open(file_path+"log.txt", 'a') as f:
        # write to the file
        f.write(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}\n')
    
    print(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}')

@dataclass
class ClassificationDataset(torch.utils.data.Dataset):
    """Wrapper on top of torch.utils.data.Dataset for differentiating test, train, val datasets."""
    
    X: torch.Tensor
    y: np.ndarray
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    
    def __getitem__(self, idx):
        
        image = self.transform(self.X[idx]) if self.transform else self.X[idx]
        label = self.target_transform(self.y[idx]) if self.target_transform else self.y[idx]
        
        return image, label

    def __len__(self):
        return len(self.X)

@dataclass
class UnsupervisedDataset(torch.utils.data.Dataset):
    """Wrapper on top of torch.utils.data.Dataset for differentiating test, train, val datasets."""
    
    image_paths: List[str]
    transform: Optional[Callable] = None,
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = image_path.split("/")[-2]
        x = Image.open(image_path)
        image = self.transform(x) if self.transform else x        
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def sample(self,):
        pass
    
    
    
class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.CNN4Backbone(
            hidden_size=hid_dim,
            channels=x_dim,
            max_pool=True,
       )
        self.out_channels = 1600

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
def mle(sample_data, init_params=[0, 1], method='Nelder-Mead'):
    if len(sample_data) <=1: return sample_data[0], 0
    sample_data = np.array(sample_data)
    mean= np.mean(sample_data)
    std= np.std(sample_data)
    
    normal= (sample_data - mean) / std

    def mle_norm(params):
        mean = params[0]   
        sd = params[1]

        # Calculate negative log likelihood
        nll = -np.sum(stats.norm.logpdf(normal, loc=mean, scale=sd))

        return nll

    results = minimize(mle_norm, init_params, method=method)
    mean, sd = (np.round((results.x * std) + mean, decimals = 4))
    return mean, 1.96*sd/sample_data.shape[0]

class PrototypicalNetwork(nn.Module):
    @staticmethod
    def pairwise_distances_logits(a, b):
        n = a.shape[0]
        m = b.shape[0]
        d = a.shape[1]
        logits = -((a.unsqueeze(1).expand(n, m, d) -
                    b.unsqueeze(0).expand(n, m, d))**2).sum(dim=2)
        return logits
    
    @staticmethod
    def accuracy(predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return ((predictions == targets).sum().float() / targets.size(0)).cpu().numpy()
    
    def __init__(self, encoder: nn.Module= Convnet(),):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
        self.hist = {"best_acc":0, "test_labels":[], "test_logits":[],
                     "val_labels":[], "val_logits":[]}
        
    def __reset_hist__(self):
        self.hist["test_labels"]=[]
        self.hist["test_logits"]=[]
        self.hist["val_labels"]=[]
        self.hist["val_logits"]=[]
    
    def data_loader(self, dataset: torch.utils.data.Dataset, way: int, shot: int, query: int, mode:str):
        def get_num_tasks(y):
            return max(y)
        
        mode_dataset = l2l.data.MetaDataset(dataset)
        mode_transforms = [
            # Samples N random classes per task
            NWays(mode_dataset, way),
            # Samples K samples per class from the above N classes (here, K = 1)
            KShots(mode_dataset, query + shot),
            # Loads a sample from the dataset
            LoadData(mode_dataset),
            # # Remaps labels starting from zero
            # RemapLabels(mode_dataset, shuffle=True),
            # Re-orders samples s.t. they are sorted in consecutive order 
            # ConsecutiveLabels(mode_dataset),
            # Randomly rotate sample over x degrees (only for vision tasks)
            RandomClassRotation(mode_dataset, [0, 90, 180, 270]),
        ]
        
        mode_tasks = l2l.data.TaskDataset(mode_dataset, task_transforms=mode_transforms,
                                          num_tasks= get_num_tasks(dataset.y) if mode!="train" else -1)
        mode_loader = DataLoader(mode_tasks, pin_memory=True, shuffle=True)
        
        return mode_loader
    
    def forward(self, support_set, support_classes, query_set, ways, shot, query_num, metric=None, device=None) -> torch.Tensor:
        if metric is None:
            metric = PrototypicalNetwork.pairwise_distances_logits
        if device is None:
            device = model.device()
        
        support_embeddings = torch.flatten(self.encoder(support_set), start_dim=1)
        query_embeddings = torch.flatten(self.encoder(query_set), start_dim=1)
        
        # print(f"support_embeddings:{support_embeddings.shape}, query_embeddings:{query_embeddings.shape}")
        centriods = support_embeddings.reshape(ways, shot, -1).mean(dim=1)

        logits = metric(query_embeddings, centriods)
        logits = torch.exp(logits) / torch.exp(logits).sum(dim=1).unsqueeze(1)
        
        return logits
    
    # def save_hist(self, X: np.ndarray, y: np.ndarray, i: int) -> None:
    #     """Saves the parameters of the model in between updates of training."""
    #     pass
    
    def save_classification_report(self, mode: str, output_path: str, epoch:int):
        report = classification_report(self.hist[f"{mode}_labels"], self.hist[f"{mode}_logits"], output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(f"{output_path}/{mode}{epoch}.csv")
        return report
    
    def save_best_model(self, n_acc, n_loss, output_path, epoch):
        loss=round(np.mean(n_loss), 3)
        loss_var= round(np.var(n_loss), 4)
        acc=round(np.mean(n_acc), 3)
        acc_var= round(np.var(n_acc), 4)

        # print(acc,  self.hist["best_acc"])
        if acc > self.hist["best_acc"]:
            self.hist["loss"] = loss
            self.hist["loss_var"] = loss_var
            self.hist["best_acc"] = acc
            self.hist["acc_var"] = acc_var
            # print(acc, self.hist["best_acc"])

            torch.save(self, output_path+ f"Epoch {epoch}-{self.hist['best_acc']}_model.pt")
            log(f"best acc: {(self.hist['best_acc'], self.hist['acc_var'])}, loss: {(self.hist['loss'], self.hist['loss_var'])}")
    
    @staticmethod
    def get_batch(data_loader, device):
        # sample data
        data, labels = next(iter(data_loader))
        data, labels = data.to(device).squeeze(0), labels.to(device).squeeze(0)
        # Sort data samples by labels
        # TODO: Use Transforms
        sort = torch.sort(labels)
        data = data[sort.indices].squeeze(0)
        labels = labels[sort.indices].squeeze(0)
        
        return data, labels
    

    @staticmethod
    def set_consecutive_labels(embbedings, classes):
        label_mapping = list(set(classes.cpu().numpy()))
        remap_list = [label_mapping.index(x) for x in classes]
        remap_classes = torch.tensor(remap_list, device=classes.device, dtype=classes.dtype)
        sort_indices = torch.sort(remap_classes)
        remap_embbedings = embbedings.squeeze(0)[sort_indices.indices].squeeze(0)
        remap_classes = remap_classes.squeeze(0)[sort_indices.indices].squeeze(0)

        return remap_embbedings, remap_classes, label_mapping


    @staticmethod
    def extract_sample(data_loader, ways, shot, query_num, device):
        data, labels = PrototypicalNetwork.get_batch(data_loader, device)

        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + query_num)
        for offset in range(shot):
            support_indices[selection + offset] = True

        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        support_set = data[support_indices]
        query_set = data[query_indices]
        support_classes = labels[support_indices]
        query_classes = labels[query_indices]

        return support_set, support_classes, query_set, query_classes
        
    
    def eval_model(self, way, shot, query, num_episode, data_loader, optimizer, metric, criterion: nn.modules.loss, device, mode, output_path, epoch=None):
        n_loss = []
        n_acc = []
        with tqdm(range(num_episode), unit="episode") as tepoch:
            if epoch != None: tepoch.set_description(f"Epoch {epoch}")
            for i in tepoch:
                batch = PrototypicalNetwork.extract_sample(data_loader= data_loader, ways= way,
                                                           shot= shot, query_num=query, device=device)
                # if i<= 3: print(batch[1], batch[3])
                support_set, support_classes, query_set, query_classes = batch
                # print(f'support_set:{support_set.shape}, support_classes:{support_classes.shape}, query_set:{query_set.shape}, query_classes:{query_set.shape}')

                remap_support_set, remap_support_classes, support_mapping = PrototypicalNetwork.set_consecutive_labels(support_set, support_classes)
                remap_query_set, remap_query_classes, query_mapping = PrototypicalNetwork.set_consecutive_labels(query_set, query_classes)
                
                logits = self.forward(support_set= remap_support_set, support_classes= remap_support_classes, query_set=remap_query_set,
                                       ways=way, shot= shot, query_num= query, metric=metric, device=device)
                
                loss = criterion(logits, remap_query_classes.long())
                acc = PrototypicalNetwork.accuracy(logits, remap_query_classes.long())

                n_loss.append(loss.item())
                n_acc.append(acc)
                # mean_loss, std_loss = mle(n_loss, init_params=[0, 1], method='Nelder-Mead')
                # mean_acc, std_acc = mle(n_acc, init_params=[0, 1], method='Nelder-Mead')

                # tepoch.set_postfix(mode=mode,
                #                    loss=f"({mean_loss:.4f}, {std_loss:.4f})",
                #                    acc=f"({mean_acc:.4f},{std_acc:.4f})")
                # if round(np.var(n_loss), 4) == np.nan: return support_set, support_classes, query_set, query_classes
                tepoch.set_postfix(mode=mode, loss=f"({round(np.mean(n_loss), 4):.4f}, {round(1.96*np.std(n_loss)/len(n_loss), 4):.4f})",
                                   acc=f"({np.round(np.mean(n_acc),decimals = 3):.3f}, {np.round(1.96*np.std(n_acc)/len(n_acc),decimals = 3):.3f})")


                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    self.hist[f"{mode}_labels"].extend([query_mapping[int(x)] for x in remap_query_classes])
                    self.hist[f"{mode}_logits"].extend([query_mapping[int(x)] for x in np.argmax(logits.cpu().detach().numpy(), axis=1)])
                
            log(f'{f"[Epoch {epoch}]"if epoch!=None else ""}[{mode}]: acc= ({np.round(np.mean(n_acc),decimals = 3):.3f}, {np.round(1.96*np.std(n_acc)/len(n_acc),decimals = 3):.3f}), loss= ({round(np.mean(n_loss), 4):.4f}, {round(1.96*np.std(n_loss)/len(n_loss), 4):.4f})', file_path=output_path)
            if mode == "test": 
                self.save_best_model(n_acc, n_loss, output_path, epoch)    
                    
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
            self.eval_model(way= train_way, shot= train_shot, query= train_query, 
                            num_episode=train_num_episode, data_loader= train_loader, optimizer=optimizer,
                            metric=None, criterion= loss, device=device, mode="train", output_path=output_path, epoch=epoch)
            # print(f'train: epoch {epoch}, loss={acc:.4f} acc={loss:.4f}')
            
            if epoch % eval_step == 0:
                
                self.eval()
                self.__reset_hist__()
                self.eval_model(way= test_way, shot= test_shot, query=test_query,
                                num_episode=test_num_episode, data_loader= val_loader, optimizer=None,
                                metric=None, criterion= loss, device=device, mode="val", output_path=output_path, epoch=epoch)
                self.save_classification_report(mode="val", output_path=output_path, epoch=epoch)
                # print(classification_report(self.hist["val_labels"], self.hist["val_logits"]))

                self.eval_model(way= test_way, shot= test_shot, query=test_query,
                        num_episode=test_num_episode, data_loader= test_loader, optimizer=None,
                        metric=None, criterion= loss, device=device, mode="test", output_path=output_path, epoch=epoch)
                self.save_classification_report(mode="test", output_path=output_path, epoch=epoch)

                # print(classification_report(self.hist["test_labels"], self.hist["test_logits"]))
                
                # print(f'eval: epoch {epoch}, loss={acc:.4f} acc={loss:.4f}')
            lr_scheduler.step()
            # if flag != None: return flag


        
        # self.eval()
        # self.__reset_hist__()
        # self.eval_model(way= test_way, shot= test_shot, query=test_query,
        #                 num_episode=test_num_episode, data_loader= test_loader, optimizer=None,
        #                 metric=None, criterion= loss, device=device, mode="test")
        # # print(classification_report(self.hist["test_labels"], self.hist["test_logits"]), output_path=output_path, epoch=epoch)
        
        print(f"best acc: {(self.hist['best_acc'], self.hist['acc_var'])}, loss: {(self.hist['loss'], self.hist['loss_var'])}")
    


# In[4]:

if __name__ == "__main__":
    pass
    
#     def init_parsearges(self):
#         """This function is for changing basic arguments of the class in training script."""
        
#         help_dict = {
#             'epochs': "number of epochs for training", 'num_episode':"number of episode per epoch",
#             'eval_step': "number of epochs before each evaluation", 'early_stop': "stop training in case overfitting occurs",
#             'gpu': "", 'train_way': "number of training classes", 'train_shot': "number of samples per class for training",
#             'train_query':"number of images to classify in training phase", 'test_way': "number of test classes",
#             'test_shot': "number of samples per class for testing and validation phase",
#             'test_query': "number of images to classify in testing and validation phase",
#         }
        
#         arg_dict = self.__dict__
#         parser = argparse.ArgumentParser()
#         for key in help_dict.keys():
#             parser.add_argument('--'+key, default=arg_dict[key],
#                                help= help_dict[key])
#         kargs = parser.parse_args()
#         kargs = {key: kargs.__dict__.get(key, arg_dict[key]) for key in arg_dict}
#         self.__init__(**kargs)
    
#     def is_trainable(self) -> (bool, List[str]):
#         training_attributes = ["epochs", "eval_step", "early_stop", "optimizer", "lr_scheduler", "loss"]
#         log = [f"argument {key} is not set" for key in training_attributes if self.__dict__[key] == None ]
        
#         return True if log==[] else False, log
    
#     def init_training(train_way: int, train_shot: int, train_query: int, train_num_episode: int,
#                 test_way: int, test_shot: int, test_query: int, test_num_episode: int, epochs: Optional[int] = None,
#                 eval_step: Optional[int] = None, early_stop: Optional[bool] = None,):

#         # training few shot learning setting
#         self.train_way= train_way
#         self.train_shot= train_shot
#         self.train_query= train_query
#         self.train_num_episode= train_num_episode

#         # testing and validation few shot learning setting
#         self.test_way= test_way
#         self.test_shot= test_shot
#         self.test_query= test_query
#         self.test_num_episode= test_num_episode

#         # general machine learning training settings
#         self.epochs= epochs
#         self.eval_step= eval_step
#         self.early_stop= early_stop
        
#         # training history
#         self.hist = None,



