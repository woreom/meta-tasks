#!/usr/bin/env python
# coding: utf-8

# In[12]:
from __future__ import annotations
import os
import argparse
import pickle
from functools import partial
from dataclasses import dataclass, field
from abc import ABC, abstractclassmethod
from typing import Dict, List, Tuple, Optional
from collections.abc import Callable, Iterable, Generator

import gdown
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

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
class FewShotLearningModel(ABC):
    """Base Representation of a FewShot Learning Model."""
    
    # training few shot learning setting
    train_way: int
    train_shot: int
    train_query: int
    
    # testing and validation few shot learning setting
    test_way: int
    test_shot: int
    test_query: int
    
    gpu: int
    
    # general machine learning training settings
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    eval_step: Optional[int] = None 
    early_stop: Optional[bool] = None
    
    optimizer: Optional[Iterable[torch.nn.parameter.Parameter]] = None
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    loss: Optional[Callable[[torch.Tensor, torch.Tensor, str, Optional[torch.Tensor],
                             Optional[bool], Optional[bool], Optional[float], Optional[int],
                             Optional[bool], Optional[bool], Optional[torch.Tensor],],
                            torch.Tensor,]] = None
    
    # training history
    hist: Optional[Dict[str, List]] = None
    
    
    def init_parsearges(self) -> None:
        """This function is for changing basic arguments of the class in training script."""
        
        help_dict = {
            'epochs': "number of epochs for training", 'batch_size': "batch size for training", 'eval_step': "number of epochs before each evaluation",
            'early_stop': "stop training in case overfitting occurs", 'gpu': "", 'train_way': "number of training classes",
            'train_shot': "number of samples per class for training", 'train_query':"number of images to classify in training phase",
            'test_way': "number of test classes", 'test_shot': "number of samples per class for testing and validation phase",
            'test_query': "number of images to classify in testing and validation phase",
        }
        
        arg_dict = self.__dict__
        parser = argparse.ArgumentParser()
        for key in help_dict.keys():
            parser.add_argument('--'+key, default=arg_dict[key],
                               help= help_dict[key])
        kargs = parser.parse_args()
        kargs = {key: kargs.__dict__.get(key, arg_dict[key]) for key in arg_dict}
        self.__init__(**kargs)
    
    def is_trainable(self) -> (bool, List[str]):
        training_attributes = ["epochs", "batch_size", "eval_step", "early_stop", "optimizer", "lr_scheduler", "loss"]
        log = [f"argument {key} is not set" for key in training_attributes if self.__dict__[key] == None ]
        
        return True if log==[] else False, log
        
    
    @abstractclassmethod
    def predict(self, X: np.ndarray) -> None:
        """Predicts the out come of Matrix X."""
        pass
    
    @abstractclassmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractclassmethod
    def load(self, path: str) -> None:
        pass
    
    @abstractclassmethod
    def data_loader(self, dataset: torch.utils.data.Dataset, mode: str):
        pass
    
    @abstractclassmethod
    def update_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Updates the weights of the model after one step or epoch."""
        pass
    
    @abstractclassmethod
    def save_hist(self, X: np.ndarray, y: np.ndarray, i: int) -> None:
        """Saves the parameters of the model in between updates of training."""
        pass
    
    @abstractclassmethod
    def fit(self, X: np.ndarray, y: np.ndarray ) -> None:
        """trains the model using the vector features X and labels Y."""
        pass
    
    # to be removed after writing prototypical network and my network
    @abstractclassmethod
    def meta_fit(model, batch, ways, shot, query_num, metric=None, device=None):
        pass
    
    @abstractclassmethod
    def encoding_fit():
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
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

    
def download_from_gdrive(_id: str, output: str) -> None:
    gdown.download(id= _id, output= output)


def get_mini_magenet(root: str, mode: str, transform: Optional[Callable] = None, download: bool= False,
                     target_transform: Optional[Callable] = None) -> Dataset:
    """Mini ImageNet Dataset Loader for torch"""
    
    gdrive_id = {"test":'1wpmY-hmiJUUlRBkO9ZDCXAcIpHEFdOhD', "train":'1I3itTXpXxGV68olxM5roceUMG8itH9Xj',
                 "val":'1KY5e491bkLFqJDp0-UWou3463Mo8AOco'}
    
    pickle_file = os.path.join(root, f'mini-imagenet_{mode}.pkl')
    
    # download pkl file
    if not os.path.exists(pickle_file) and download:
        print(f"Downloading mini-imagenet_{model} at {pickle_file}")
        download_from_gdrive(_id=MiniImagenet.gdrive_id[mode], output=pickle_file)
        
    # open pkl file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # create X and y
    X = torch.from_numpy(data["image_data"]).permute(0, 3, 1, 2).float()
    y = np.ones(X.shape[0])

    # dict of indexes to np.array
    for (i, key) in enumerate(data['class_dict'].keys()):
        for idx in data['class_dict'][key]:
            y[idx] = i

    return ClassificationDataset(X= X, y= y, transform= transform,
                                 target_transform= target_transform)

@dataclass
class PrototypicalNetwork(FewShotLearningModel):
    encoder: nn.Module= Convnet()
    optimizer: Optional[Iterable[torch.nn.parameter.Parameter]]= torch.optim.Adam
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.StepLR
    loss: Optional[Callable[[torch.Tensor, torch.Tensor, str, Optional[torch.Tensor],
                             Optional[bool], Optional[bool], Optional[float], Optional[int],
                             Optional[bool], Optional[bool], Optional[torch.Tensor],],
                            torch.Tensor,]] = F.cross_entropy
    
    def save(self, path: str) -> None:
        pass
    
    def load(self, path: str) -> None:
        pass
    
    def predict(self, X: np.ndarray) -> None:
        """Predicts the out come of Matrix X."""
        pass
    
    def data_loader(self, dataset: torch.utils.data.Dataset, mode: str):
        def get_num_tasks(y):
            return max(y)
        
        mode_dataset = l2l.data.MetaDataset(dataset)
        mode_transforms = [
            NWays(mode_dataset, self.train_way if mode == "train" else self.test_way),
            KShots(mode_dataset, self.train_query + self.train_shot if mode == "train" else self.test_query + self.test_shot),
            LoadData(mode_dataset),
            RemapLabels(mode_dataset),
        ]
        
        mode_tasks = l2l.data.TaskDataset(mode_dataset, task_transforms=mode_transforms,
                                          num_tasks= get_num_tasks(dataset.y) if mode!="train" else -1)
        mode_loader = DataLoader(mode_tasks, pin_memory=True, shuffle=True)
        
        return mode_loader
    
    def meta_fit(model, batch, ways, shot, query_num, metric=None, device=None):
        pass
    
    def encoding_fit():
        pass
    
    def update_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Updates the weights of the model after one step or epoch."""
        pass
    
    def save_hist(self, X: np.ndarray, y: np.ndarray, i: int) -> None:
        """Saves the parameters of the model in between updates of training."""
        pass
    
    def fit(self, get_dataset: Callable,
            optimizer: Optional[Iterable[torch.nn.parameter.Parameter]]= None,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]= None,
            loss: Optional[Callable[[torch.Tensor, torch.Tensor, str, Optional[torch.Tensor],
                                    Optional[bool], Optional[bool], Optional[float], Optional[int],
                                    Optional[bool], Optional[bool], Optional[torch.Tensor],
                                    ], torch.Tensor,]]= None,
            ) -> None:
        
        # print("aaa")
        trainable, log = self.is_trainable()
        if not trainable:
            raise Exception(log[0])
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(device)
        
        optimizer = self.optimizer(self.encoder.parameters(), lr=0.001)
        lr_scheduler = self.lr_scheduler(optimizer, step_size=20, gamma=0.5)
        
        train_loader = self.data_loader(dataset= get_dataset(mode= 'train'), mode= 'train')
        val_loader = self.data_loader(dataset= get_dataset(mode= 'val'), mode= 'val')
        test_loader = self.data_loader(dataset= get_dataset(mode= 'test'), mode= 'test')
        
        

        for epoch in range(1, self.epochs + 1):
            # print("aaa")
            self.encoder.train()

            loss_ctr = 0
            n_loss = 0
            n_acc = 0

            for i in range(100):
                batch = next(iter(train_loader))

                loss, acc = fast_adapt(self.encoder,
                                       batch,
                                       self.train_way,
                                       self.train_shot,
                                       self.train_query,
                                       metric=pairwise_distances_logits,
                                       device=device)

                loss_ctr += 1
                n_loss += loss.item()
                n_acc += acc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

            print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
                epoch, n_loss/loss_ctr, n_acc/loss_ctr))

            self.encoder.eval()

            loss_ctr = 0
            n_loss = 0
            n_acc = 0
            for i, batch in enumerate(val_loader):
                loss, acc = fast_adapt(self.encoder,
                                       batch,
                                       self.test_way,
                                       self.test_shot,
                                       self.test_query,
                                       metric=pairwise_distances_logits,
                                       device=device)

                loss_ctr += 1
                n_loss += loss.item()
                n_acc += acc

            print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
                epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        loss_ctr = 0
        n_acc = 0

        for i, batch in enumerate(test_loader, 1):
            loss, acc = fast_adapt(self.encoder,
                                   batch,
                                   self.test_way,
                                   self.test_shot,
                                   self.test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)
            loss_ctr += 1
            n_acc += acc
            print('batch {}: {:.2f}({:.2f})'.format(
                i, n_acc/loss_ctr * 100, acc * 100))

def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


model = PrototypicalNetwork(encoder= Convnet(), epochs= 250, train_way= 30, train_shot= 1, train_query= 15,
                            test_way= 5, test_shot= 1, test_query= 30, gpu= 1, batch_size=8,
                            eval_step= 10, early_stop= True,)
model.init_parsearges()

print(model)

get_dataset = partial(get_mini_magenet, root="../../datasets/mini_imagenet/", transform=None, target_transform=None, download=False)

model.fit(get_dataset=get_dataset)


