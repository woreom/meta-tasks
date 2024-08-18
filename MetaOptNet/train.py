# -*- coding: utf-8 -*-
import os
import argparse
from distutils.util import strtobool

import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms


# import datasets
from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

from utils import set_gpu, Timer, count_accuracy, check_dir, log


state = 42
torch.manual_seed(state)
torch.cuda.manual_seed(state)
np.random.seed(state)
random.seed(state)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels: int, output_channels: int, upsample: bool):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3,
                                   stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3,
                          stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1,
                                   stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x
    
class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels: int, output_channels: int, layers: int):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels,
                                             upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels,
                                             upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for name, layer in self.named_children():

            x = layer(x)

        return x

class ResNet12Decoder(nn.Module):
    def __init__(self, kernel_size=7,):
        super(ResNet12Decoder, self).__init__()
        configs = [1, 2, 2, 2]
        # self.linear = nn.Linear(in_features=512, out_features=512*3*3)
        self.conv1 = DecoderResidualBlock(hidden_channels=640, output_channels=320, layers=configs[0])
        self.conv2 = DecoderResidualBlock(hidden_channels=320, output_channels=160, layers=configs[1])
        self.conv3 = DecoderResidualBlock(hidden_channels=160, output_channels=64,  layers=configs[2])
        # self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=kernel_size, stride=2, padding=1,
                               output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class SimpleAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, reshape_size=(-1,640,5,5)):
        super(SimpleAutoEncoder, self).__init__()
        self.reshape_size = reshape_size
        
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(self.reshape_size)
        x = self.decoder(x)
        return x


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=[int(i) for i in opt.gpu.replace(" ","").split(',')])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--meta-task', dest='meta_task', 
                            type=lambda x: bool(strtobool(x)),
                            help='use meta-task regularization')
    parser.add_argument('--meta-task-lr', type=float, default=5e-4,
                            help='learning rate for meta-task')
    parser.add_argument('--meta-task-lr-scheduler', dest='meta_task_lr_scheduler', 
                            type=lambda x: bool(strtobool(x)),
                            help='use meta-task regularization')
    
    opt = parser.parse_args()
    
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000, # num of batches per epoch
    )
    
    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load('./experiments/tieredImageNet_MetaOptNet_meta-task/Epoch_211.pth')

    try:
        embedding_net.load_state_dict(saved_models['embedding'])
        embedding_net.eval()
        cls_head.load_state_dict(saved_models['head'])
        cls_head.eval()
    
    except Exception as e:

        new_state_dict = OrderedDict()
        new_state_dict['embedding'] = OrderedDict()
        for k, v in saved_models['embedding'].items():
            name = k[7:] # remove `module.`
            new_state_dict['embedding'][name] = v

        new_state_dict['head'] = saved_models['head']
        
        embedding_net.load_state_dict(new_state_dict['embedding'])
        embedding_net.eval()
        cls_head.load_state_dict(new_state_dict['head'])
        cls_head.eval()
    
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    # lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lambda_epoch = lambda e: 0.012
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    # meta-task autoencoder
    if opt.meta_task:
        kernel_size = 7 if opt.dataset == 'miniImageNet' or "tieredImageNet" else 3
        reshape_size = (-1,640,5,5) if opt.dataset == 'miniImageNet' or "tieredImageNet" else (-1,640,2,2)
        autoencoder = SimpleAutoEncoder(embedding_net, ResNet12Decoder(kernel_size=kernel_size), reshape_size=reshape_size).cuda()
        autoencoder.decoder = torch.nn.DataParallel(autoencoder.decoder, device_ids=[int(i) for i in opt.gpu.replace(" ","").split(',')])
        criterion = nn.MSELoss()

        optim = torch.optim.Adam(autoencoder.parameters(),
                                    lr=opt.meta_task_lr)
        if opt.meta_task_lr_scheduler:
            lambda_epoch = lambda e: 1.0 if e < 20 else 0.5 if e < 40 else 0.05
            meta_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(211, opt.num_epoch + 211):
        # Train on the training split
        lr_scheduler.step()
        if opt.meta_task_lr_scheduler and opt.meta_task: meta_scheduler.step()
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))

            if opt.meta_task:
                data = torch.cat((data_support, data_query),0)
                recon = autoencoder(data)
                error = criterion(recon, data)

                optim.zero_grad()
                error.backward()
                optim.step()

            emb_support = embedding_net(data_support)
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            
            emb_query = embedding_net(data_query)
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
            
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            
            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
            
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, f'best_model_{val_acc_avg}.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, f'Epoch_{epoch}.pth'))
        
        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
