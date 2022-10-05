import torch
import torch.nn as nn

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


class VGG16(nn.Module):
    def __init__(self, net_name):
        super(VGG16, self).__init__()
        self.feature_extraction = self.create_layers(cfg[net_name])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x

    def create_layers(self, cfg):
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        return nn.Sequential(*layers)
