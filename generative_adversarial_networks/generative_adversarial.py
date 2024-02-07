import os
from typing import Callable

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, num_channels=1, input_size=100, feature_size=64, num_gpu=1):
        super(Generator, self).__init__()
        self.ngpu = num_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_size, feature_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature_size * 8, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( feature_size * 4, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( feature_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self, num_channels=1, feature_size=64, num_gpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = num_gpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_channels, feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(feature_size * 2, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(feature_size * 4, feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels= feature_size * 8,
                      out_channels= 1, kernel_size= 4,
                      stride=1,padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
class GANSupervisor:

    def random_noise(self, size):
        return torch.randn(size, self.embedding_size, 1, 1, device=self.device)

    def __init__(self, generator: nn.Module, discriminator: nn.Module,
                 generator_optimizer: optim, discriminator_optimizer: optim,
                 generator_learning_rate: float, discriminator_learning_rate: float,
                 get_dataset: Callable, get_noise_generator: Callable,
                 epoch: int, batch_size:int, embedding_size: int) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gen = generator
        self.gen.to(self.device)
        self.gen_lr = generator_learning_rate
        self.embedding_size = embedding_size
        self.gen_opt = generator_optimizer

        beta1 = 0.5
        self.gen_optim = self.gen_opt(self.gen.parameters(), lr=self.gen_lr, betas=(beta1, 0.999))
        
        self.dis = discriminator
        self.dis.to(self.device)
        self.dis_opt = discriminator_optimizer
        self.dis_lr = discriminator_learning_rate
        self.dis_optim= self.dis_opt(self.dis.parameters(), lr=self.dis_lr, betas=(beta1, 0.999))

        self.epoch = epoch
        self.batch_size = batch_size

        self.get_dataset = get_dataset
        self.train_dataset = self.get_dataloader(mode= "train")
        # self.val_dataset = self.get_dataloader(mode= "val")
        self.test_dataset = self.get_dataloader(mode= "test")

        self.hist={"epoch":[], "G_loss":[], "D_Loss":[], "real_loss":[], "fake_loss":[],}
        self.make_noise = get_noise_generator if get_noise_generator != None else self.random_noise


        

    def get_dataloader(self, mode:str):

        dataset = self.get_dataset(mode=mode)
        # print(len(dataset))
        loader = torch.utils.data.DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True,)
        # print(len(loader))

        
        return loader

    def train_generator(self, X):
        self.gen.train()
        self.gen.zero_grad()

        batch_size = X.size(0)
        ones_label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        noise = self.make_noise(batch_size)
        # Generate fake image batch with G
        fake_samples = self.gen(noise)
        # print(fake_samples.shape)
        # exit
        D_fake = self.dis(fake_samples).view(-1)
        # Calculate G's loss based on this output
        G_loss = self.criterion(D_fake, ones_label)
        # Calculate gradients for G
        G_loss.backward()
        # Update G
        self.gen_optim.step()

        return G_loss.item()


    def train_discriminator(self, X):


        self.dis.train()
        self.dis.zero_grad()
        batch_size = X.size(0)
        ones_label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)
        # Forward pass real batch through D
        D_real = self.dis(X).view(-1)
        # Calculate loss on all-real batch
        # print(f"D_real: {D_real.shape}", f"ones_label: {ones_label.shape}")
        D_loss_real = self.criterion(D_real, ones_label)
        # Calculate gradients for D in backward pass
        D_loss_real.backward()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = self.make_noise(batch_size)
        # Generate fake image batch with G
        fake_samples = self.gen(noise)
        zeros_label = torch.full((batch_size,), 0, dtype=torch.float, device=self.device)
        # Classify all fake batch with D
        D_fake = self.dis(fake_samples.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        D_loss_fake = self.criterion(D_fake, zeros_label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        D_loss_fake.backward()
        # Compute error of D as sum over the fake and the real batches
        D_loss = D_loss_real + D_loss_fake
        # Update D
        self.dis_optim.step()

        return D_loss.item(), D_loss_real.item(), D_loss_fake.item()
    
    def fit(self) -> None:
        fixed_noise = self.make_noise(16)
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()     
        for epoch in range(self.epoch):
            G_losses = [0]
            D_losses = [0]
            real_losses = [0]
            fake_losses = [0] 
            # For each batch in the dataloader
            
            with tqdm(range(len(self.train_dataset)), unit="batch") as batch:
                if epoch != None: batch.set_description(f"Epoch {epoch}")
                for i in batch:
                    data = next(iter(self.train_dataset))
                    # Sample data
                    X = data[0].to(self.device)

                    if G_losses[-1] - D_losses[-1] >= 1.5:
                        G_loss = self.train_generator(X)

                    elif G_losses[-1] - D_losses[-1] <= -1.5:
                        D_loss, real_loss, fake_loss = self.train_discriminator(X)
                    
                    else:
                        D_loss, real_loss, fake_loss = self.train_discriminator(X)
                        G_loss = self.train_generator(X)
                        
                    G_losses.append(G_loss)
                    D_losses.append(D_loss)
                    real_losses.append(real_loss)
                    fake_losses.append(fake_loss)

                    batch.set_postfix(G_loss=f"{round(np.mean(G_losses), 4):.4f}", D_loss=f"{round(np.mean(D_losses), 4):.4f}",
                                   real_loss=f"{np.round(np.mean(real_losses),decimals = 3):.3f}", fake_loss = f"{np.round(np.mean(fake_losses),decimals = 3):.3f}")

            self.save_hist(epoch, G_losses, D_losses, real_losses, fake_losses)
            if (epoch % (self.epoch//100) == 0) or (epoch == self.epoch-1):
                self.save_sample(fixed_noise=fixed_noise, filename=f"{epoch}", output_dir="out2/")
                torch.save(self.gen, f"out2/gen/gen{epoch}.pt")
                torch.save(self.dis, f"out2/dis/dis{epoch}.pt")

        self.plot_hist()

    def save_sample(self, fixed_noise, output_dir, filename):
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            # Generate fake image batch with G
            fake_samples = self.gen(fixed_noise)
            fake = fake_samples.detach().cpu().numpy()

        fig= plt.figure(figsize=(4, 4))
        fig.suptitle(filename)

        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.axis("off")
            plt.imshow(fake[i].reshape((fake[i].shape[1],fake[i].shape[2], fake[i].shape[0])))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(f'{output_dir}/{filename}.png', bbox_inches='tight')
        plt.close(fig)

    def save_hist(self, epoch, G_losses, D_losses, real_losses, fake_losses) -> None:

        self.hist["epoch"].append(epoch)
        self.hist["G_loss"].append(G_losses[-1])
        self.hist["D_Loss"].append(D_losses[-1])
        self.hist["real_loss"].append(real_losses[-1])
        self.hist["fake_loss"].append(fake_losses[-1])

    def plot_hist(self):
        pass
