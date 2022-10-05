import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from model import VGG16

bs = 128
num_epochs = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = dsets.CIFAR10(root='../data/CIFAR10', train=True,
                              transform=transform, download=True)
test_dataset = dsets.CIFAR10(root='../data/CIFAR10', train=False,
                             transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = VGG16("VGG16").to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train():
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (imgs, label) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            imgs = imgs.to(device)
            label = label.to(device)

            label_predicted = model(imgs)
            loss = criterion(label_predicted, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(
            "[Epoch %d/%d] [train loss: %f]"
            % (epoch + 1, num_epochs, train_loss)
        )

train()