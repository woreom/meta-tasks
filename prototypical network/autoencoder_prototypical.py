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

# impelementations
from prototypical_network import ClassificationDataset, PrototypicalNetwork, Convnet

def log(*args, file_path: str=''):
    now = datetime.now()
    # open the file in append mode
    with open(file_path+"log.txt", 'a') as f:
        # write to the file
        f.write(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}\n')
    
    print(f'[{now.strftime("%d/%m/%Y %H:%M:%S")}]{" ".join(map(str,args))}')

class PrototypicalAutoencoderNetwork(PrototypicalNetwork):
    
    def __init__(self, autoencoder: nn.modules):
        super(PrototypicalAutoencoderNetwork, self).__init__()
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.update = False
        
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
        
        return logits

    def init_encoder_optimizer(self, autoencoder_lr=1e-6, autoencoder_weight_decay= 1e-8):
        self.autoencoder.loss = nn.MSELoss()
        self.autoencoder.optimizer = torch.optim.Adam(self.autoencoder.parameters(),
                             lr=autoencoder_lr, 
                             weight_decay=autoencoder_weight_decay)
    
    def update_encoder(self, support_set, query_set, device):
        self.autoencoder.train()
        batch = torch.cat((support_set, query_set), 0)
        batch = batch.to(device)

        recon = self.autoencoder.forward(batch)
        loss = self.autoencoder.loss(recon, batch)

        self.autoencoder.optimizer.zero_grad()
        loss.backward()
        self.autoencoder.optimizer.step()

    
    def eval_model(self, way, shot, query, num_episode, data_loader, optimizer, metric, criterion: nn.modules.loss, device, mode, output_path, epoch=None):
        n_loss = []
        n_acc = []
        with tqdm(range(num_episode), unit="episode") as tepoch:
            if epoch != None: tepoch.set_description(f"Epoch {epoch}")
            
            for i in tepoch:
                batch = PrototypicalNetwork.extract_sample(data_loader= data_loader, ways= way,
                                                           shot= shot, query_num=query, device=device)
                
                support_set, support_classes, query_set, query_classes = batch
                # print(f'support_set:{support_set.shape}, support_classes:{support_classes.shape}, query_set:{query_set.shape}, query_classes:{query_set.shape}')

                remap_support_set, remap_support_classes, support_mapping = PrototypicalNetwork.set_consecutive_labels(support_set, support_classes)
                remap_query_set, remap_query_classes, query_mapping = PrototypicalNetwork.set_consecutive_labels(query_set, query_classes)

                if mode == "train" or self.update: self.update_encoder(remap_support_set, remap_query_set, device)
                logits = self.forward(support_set= remap_support_set, support_classes= remap_support_classes, query_set=remap_query_set,
                                       ways=way, shot= shot, query_num= query, metric=metric, device=device)
                
                loss = criterion(logits, remap_query_classes.long())
                acc = PrototypicalNetwork.accuracy(logits, remap_query_classes.long())
                
                n_loss.append(loss.item())
                n_acc.append(acc)
                
                # tepoch.set_postfix(mode=mode, loss=f"({round(np.mean(n_loss), 4):.4f}, {round(np.var(n_loss), 4):.4f})",
                #                    acc=f"({np.round(np.mean(n_acc),decimals = 3):.3f}, {np.round(np.var(n_acc),decimals = 3):.3f})")
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
            if mode == "test": self.save_best_model(n_acc, n_loss, output_path, epoch)
                    
    def fit(self, train_way: int, train_shot: int, train_query: int, train_num_episode: int,
            test_way: int, test_shot: int, test_query: int, test_num_episode: int, epochs:int, get_dataset: Callable,
            eval_step:int, early_stop:bool, optimizer: torch.optim, lr_scheduler, loss: nn.modules.loss, output_path,
            autoencoder_lr=1e-6, autoencoder_weight_decay= 1e-8) -> None:
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        train_loader = self.data_loader(dataset= get_dataset(mode= 'train'),
                                        way= train_way, shot= train_shot, query= train_query, mode='train')
        val_loader = self.data_loader(dataset= get_dataset(mode= 'val'),
                                      way= test_way, shot= test_shot, query=test_query, mode='val')
        test_loader = self.data_loader(dataset= get_dataset(mode= 'test'),
                                       way= test_way, shot= test_shot, query=test_query, mode='test')
        
        
        self.init_encoder_optimizer(autoencoder_lr=autoencoder_lr, autoencoder_weight_decay= autoencoder_weight_decay)
        for epoch in range(1, epochs+1,):
            # print("aaa")
            self.train()
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
                # print(f'eval: epoch {epoch}, loss={acc:.4f} acc={loss:.4f}')
                # print(classification_report(self.hist["val_labels"], self.hist["val_logits"]))
                self.save_classification_report(mode="val", output_path=output_path, epoch=epoch)

                self.eval()
                self.__reset_hist__()
                self.eval_model(way= test_way, shot= test_shot, query=test_query,
                        num_episode=test_num_episode, data_loader= test_loader, optimizer=None,
                        metric=None, criterion= loss, device=device, mode="test", output_path=output_path, epoch=epoch)
                self.save_classification_report(mode="test", output_path=output_path, epoch=epoch)

            lr_scheduler.step()

        
        # self.eval()
        # self.__reset_hist__()
        # self.eval_model(way= test_way, shot= test_shot, query=test_query,
        #                 num_episode=test_num_episode, data_loader= test_loader, optimizer=None,
        #                 metric=None, criterion= loss, device=device, mode="test", output_path=output_path)
        # # print(classification_report(self.hist["test_labels"], self.hist["test_logits"]))

        print(f"best acc: {(self.hist['best_acc'], self.hist['acc_var'])}, loss: {(self.hist['loss'], self.hist['loss_var'])}")
        

class LayerFitter(nn.Module):
    def __init__(self, device=None):
        super(LayerFitter, self).__init__()
        self.device = device
        
    def fit(self, X, y):
        
        self.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        
        
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                             lr=1e-1, 
                             weight_decay=1e-5)
        
        for i in range(10):
            y_pred = self(X, y)
            loss = loss_func(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
class ArcLayer(LayerFitter):
    
    def __init__(self, out_features, s=8, m=0.5, device=None):
        super().__init__()
        self.device = device
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.ways = out_features
        self.fc = nn.Linear(in_features=512*3*3, out_features=out_features, bias=False)

    def forward(self, x,  label=None):
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / (x_L2 * w_L2)
        
        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=self.ways)
            sin = (1 - cos ** 2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
        
        return cos
        
########################################################################################
# Source:  https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
########################################################################################

class ArcMarginProduct(LayerFitter):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device=None):
        super(ArcMarginProduct, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.update = False

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is not None:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            cosine = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            cosine *= self.s
            # print(output)

        return cosine


class AddMarginProduct(LayerFitter):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40, device=None):
        super(AddMarginProduct, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.update = False

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is not None:
            phi = cosine - self.m
            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cosine.size(), device='cuda')
            # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            cosine = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            cosine *= self.s
            # print(output)

        return cosine

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(LayerFitter):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4, device=None):
        super(SphereProduct, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)
        self.update = False

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label=None):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is not None:
            cos_theta = cos_theta.clamp(-1, 1)
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = cos_theta.data.acos()
            k = (self.m * theta / 3.14159265).floor()
            phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
            NormOfFeature = torch.norm(input, 2, 1)

            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cos_theta.size())
            one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1), 1)

            # --------------------------- Calculate output ---------------------------
            cos_theta = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
            cos_theta *= NormOfFeature.view(-1, 1)

        return cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

########################################################################################

    
        
class ArcAutoencoder(PrototypicalAutoencoderNetwork):
    def __init__(self, autoencoder: nn.modules, fc: nn.modules):
        super(PrototypicalAutoencoderNetwork, self).__init__()
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.fc = fc
        self.update = False
        
    def forward(self, support_set, support_classes, query_set, ways, shot, query_num, metric=None, device=None) -> torch.Tensor:
        if metric is None:
            metric = PrototypicalNetwork.pairwise_distances_logits
        if device is None:
            device = model.device()

        support_embeddings = torch.flatten(self.encoder(support_set), start_dim=1)
        query_embeddings = torch.flatten(self.encoder(query_set), start_dim=1)

        fc = self.fc(out_features= ways, device= device) 

        fc.train()
        fc.fit(support_embeddings, support_classes)

        fc.eval()
        logits = fc(query_embeddings)

        return logits
    