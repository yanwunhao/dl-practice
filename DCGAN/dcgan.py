import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Set random seed for reproducibility
manualSeed = 999

print("Random Seed: ", manualSeed)

data_root = "../data/CelebA"
img_size = 64
batch_size = 128

feature_map_of_generator = 64
feature_map_of_discriminator = 64

num_of_channels = 3

latent_dim = 100


# Set random seed to random module and
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataset = dset.ImageFolder(root=data_root,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               # transforms.Resize([img_size, img_size]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_of_generator * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_of_generator * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator * 8, feature_map_of_generator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_generator * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator * 4, feature_map_of_generator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_generator * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator * 2, feature_map_of_generator, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_generator),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator, num_of_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


generator = Generator().to(device)

generator.apply(weights_init)
print(generator)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_of_channels, feature_map_of_discriminator, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator, feature_map_of_discriminator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_discriminator * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator * 2, feature_map_of_discriminator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_discriminator * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator * 4, feature_map_of_discriminator * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_discriminator * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


discrimintor = Discriminator().to(device)
discrimintor.apply(weights_init)

print(discrimintor)