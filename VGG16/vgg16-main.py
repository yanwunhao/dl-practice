import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from model import VGG16

bs = 128
num_epochs = 100

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
        for batch_idx, (imgs, labels) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)

            labels_predicted = model(imgs)
            loss = criterion(labels_predicted, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [train loss: %f]"
                    % (epoch + 1, num_epochs, batch_idx + 1, len(train_dataloader), train_loss / 50.)
                )
                train_loss = 0

        total = 0
        correct = 0

        with torch.no_grad():
            for (imgs, labels) in test_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                labels_predicted = model(imgs)
                _, class_predicted = torch.max(labels_predicted.data, 1)
                total += labels.size(0)
                correct += (class_predicted == labels).cpu().sum().item()
            print('Accuracy on the 10000 test images: %d %% ' % ((100 * correct) / total))

train()