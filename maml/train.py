import argparse
import os
import random
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.optimizers as optimizers
from models.encoders.autoencoders import DecoderResidualBlock
from torchsummary import summary

class ResNet12Decoder(nn.Module):
    def __init__(self):
        super(ResNet12Decoder, self).__init__()
        configs = [1, 2, 2, 2]
        # self.linear = nn.Linear(in_features=512, out_features=512*3*3)
        self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
        self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
        self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=configs[2])
        self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=7,
                               output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((x.shape[0], 512, 1, 1))
        x = x.expand(x.shape[0], 512, 3, 3)
        # x = x.expand(1, 512, 3, 3)
        # x = x.reshape((x.shape[0], 512, 1, 1))
        # print(x.shape)
        # x = self.linear(x)
        # x = x.reshape((x.shape[0],-1,3,3))
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class Shallow_Decoder(torch.nn.Sequential):

    def __init__(self,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = [nn.ConvTranspose2d(in_channels=channels, out_channels=hidden, kernel_size=3,
                                   stride=2, padding=1,), nn.BatchNorm2d(hidden), nn.ReLU()]
        for _ in range(layers - 2):
            core.extend([nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=3,
                                            stride=1, padding=0,), nn.BatchNorm2d(hidden), nn.ReLU()])
            
        core.extend([nn.ConvTranspose2d(in_channels=hidden, out_channels=3, kernel_size=2,
                                        stride=1, padding=0,), nn.BatchNorm2d(3),nn.Sigmoid()])
            
        super(Shallow_Decoder, self).__init__(*core)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((x.shape[0], 1, 40, 40))
        x = super(Shallow_Decoder, self).forward(x)

        return x
    
class Deep_Decoder(torch.nn.Sequential):

    def __init__(self,
                 hidden=64,
                 channels=64,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = [nn.ConvTranspose2d(in_channels=channels, out_channels=hidden, kernel_size=3,
                                   stride=2, padding=0,), nn.BatchNorm2d(hidden), nn.ReLU()]
        for _ in range(layers - 2):
            core.extend([nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=5,
                                            stride=2, padding=2,), nn.BatchNorm2d(hidden), nn.ReLU()])
            
        core.extend([nn.ConvTranspose2d(in_channels=hidden, out_channels=3, kernel_size=6,
                                        stride=2, padding=1,), nn.BatchNorm2d(3),nn.Sigmoid()])
            
        super(Deep_Decoder, self).__init__(*core)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((x.shape[0], 64, 5, 5))
        # print(x.shape)
        # x = x.expand(x.shape[0], 1600, 3, 3)
        x = super(Deep_Decoder, self).forward(x)

        return x

class SimpleAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(SimpleAutoEncoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False

  ckpt_name = args.name
  meta_task = config.get('meta_autoencoder')
  if ckpt_name is None:
    ckpt_name = config['encoder']
    ckpt_name += '_' + config['dataset'].replace('meta-', '').replace('folder-','')+f'-{meta_task}'
    ckpt_name += '_{}_way_{}_shot'.format(
      config['train']['n_way'], config['train']['n_shot'])
  if args.tag is not None:
    ckpt_name += '_' + args.tag

  ckpt_path = os.path.join('../outputs', ckpt_name)
  utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

  ##### Dataset #####

  # meta-train
  train_set = datasets.make(config['dataset'], **config['train'])
  utils.log('meta-train set: {} (x{}), {}'.format(
    train_set[0][0].shape, len(train_set), train_set.n_classes))
  train_loader = DataLoader(
    train_set, config['train']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

  # meta-val
  eval_val = False
  if config.get('val'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val'])
    utils.log('meta-val set: {} (x{}), {}'.format(
      val_set[0][0].shape, len(val_set), val_set.n_classes))
    val_loader = DataLoader(
      val_set, config['val']['n_episode'],
      collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)
    
  # meta-test
  eval_test = False
  if config.get('test'):
    eval_test = True
    test_set = datasets.make(config['dataset'], **config['test'])
    utils.log('meta-test set: {} (x{}), {}'.format(
      test_set[0][0].shape, len(test_set), test_set.n_classes))
    test_loader = DataLoader(
      test_set, config['test']['n_episode'],
      collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)
  
  ##### Model and Optimizer #####

  inner_args = utils.config_inner_args(config.get('inner_args'))
  if config.get('load'):
    ckpt = torch.load(config['load'])
    config['encoder'] = ckpt['encoder']
    config['encoder_args'] = ckpt['encoder_args']
    config['classifier'] = ckpt['classifier']
    config['classifier_args'] = ckpt['classifier_args']
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
    optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
    start_epoch = ckpt['training']['epoch'] + 1
    max_va = ckpt['training']['max_va']
  else:
    config['encoder_args'] = config.get('encoder_args') or dict()
    config['classifier_args'] = config.get('classifier_args') or dict()
    config['encoder_args']['bn_args']['n_episode'] = config['train']['n_episode']
    config['classifier_args']['n_way'] = config['train']['n_way']
    model = models.make(config['encoder'], config['encoder_args'],
                        config['classifier'], config['classifier_args'])
    optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])
    start_epoch = 1
    max_va = 0.

    meta_task = config.get('meta_autoencoder')
    if meta_task:
      # autoencoder = SimpleAutoEncoder(model.encoder, ResNet12Decoder()).cuda()
      # autoencoder = SimpleAutoEncoder(model.encoder, Shallow_Decoder()).cuda()
      autoencoder = SimpleAutoEncoder(model.encoder, Deep_Decoder()).cuda()

      opt = torch.optim.Adam(autoencoder.parameters(),
                                  lr=1e-6)
      criterion = nn.MSELoss()

      # _input = torch.randn((1,3,84,84)).cuda()

      # print(f"encoder output: {_input.shape}")

      # output = autoencoder.encoder(_input)

      # print(f"encoder output: {output.shape}")

      # output = autoencoder.decoder(output)

      # print(f"decoder output: {output.shape}")

      # print(summary(autoencoder.encoder,(3,84,84)))
      # print(summary(autoencoder.decoder,(1600, 1, 1)))

    
  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  ##### Training and evaluation #####
    
  # 'tl': meta-train loss
  # 'ta': meta-train accuracy
  # 'vl': meta-val loss
  # 'va': meta-val accuracy
  # 'tel': meta-test loss
  # 'tea': meta-test accuracy
  aves_keys = ['tl', 'ta', 'vl', 'va', 'tel', 'tea']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  for epoch in range(start_epoch, config['epoch'] + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}

    # meta-train
    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    np.random.seed(epoch)

    for data in tqdm(train_loader, desc='meta-train', leave=False):
      x_shot, x_query, y_shot, y_query = data
      x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
      x_query, y_query = x_query.cuda(), y_query.cuda()

      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()
      if meta_task:
        img_size = x_shot.shape[-1]
        recon = autoencoder(x_shot.reshape(-1,3,img_size,img_size))
        error = criterion(recon, x_shot.reshape(-1,3,img_size,img_size))

        opt.zero_grad()
        error.backward()
        opt.step()

      logits = model(x_shot, x_query, y_shot, inner_args, meta_train=True)
      logits = logits.flatten(0, 1)
      labels = y_query.flatten()
      
      pred = torch.argmax(logits, dim=-1)
      acc = utils.compute_acc(pred, labels)
      loss = F.cross_entropy(logits, labels)
      aves['tl'].update(loss.item(), 1)
      aves['ta'].update(acc, 1)
      
      optimizer.zero_grad()
      loss.backward()
      for param in optimizer.param_groups[0]['params']:
        nn.utils.clip_grad_value_(param, 10)
      optimizer.step()

    # meta-val
    if eval_val:
      model.eval()
      np.random.seed(0)

      for data in tqdm(val_loader, desc='meta-val', leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()
        
        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)
        aves['vl'].update(loss.item(), 1)
        aves['va'].update(acc, 1)

    # meta-test
    if eval_test:
      model.eval()
      np.random.seed(0)

      for data in tqdm(test_loader, desc='meta-test', leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()
        
        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)
        aves['tel'].update(loss.item(), 1)
        aves['tea'].update(acc, 1)

    if lr_scheduler is not None:
      lr_scheduler.step()

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])

    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() / 
      (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))

    # formats output
    log_str = f"Epoch {epoch}:(train) acc={aves['ta']*100:.2f}%, loss={aves['tl']:.4f}"

    writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)

    if eval_val:
      log_str += f"|(val) acc={aves['va']*100:.2f}%, loss={aves['vl']:.4f}"
      writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
      writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

    if eval_test:
      log_str += f"|(test) acc={aves['tea']*100:.2f}%, loss={aves['tel']:.4f}"
      writer.add_scalars('loss', {'meta-test': aves['tel']}, epoch)
      writer.add_scalars('acc', {'meta-test': aves['tea']}, epoch)

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    # saves model and meta-data
    if config.get('_parallel'):
      model_ = model.module
    else:
      model_ = model

    training = {
      'epoch': epoch,
      'max_va': max(max_va, aves['va']),

      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict() 
        if lr_scheduler is not None else None,
    }
    ckpt = {
      'file': __file__,
      'config': config,

      'encoder': config['encoder'],
      'encoder_args': config['encoder_args'],
      'encoder_state_dict': model_.encoder.state_dict(),

      'classifier': config['classifier'],
      'classifier_args': config['classifier_args'],
      'classifier_state_dict': model_.classifier.state_dict(),

      'training': training,
    }

    # 'epoch-last.pth': saved at the latest epoch
    # 'max-va.pth': saved when validation accuracy is at its maximum
    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))

    writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--name', 
                      help='model name', 
                      type=str, default=None)
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config)