import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
summary(model, (1, 28, 28))

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
print(f"logits: {logits}")
pred_probab = nn.Softmax(dim=1)(logits)
print(f"Predicted probability: {pred_probab}")
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# create an image-like tensor
input_image = torch.rand(3, 28, 28)
print(f"Input size: {input_image.size()}")

# example of flatten layer
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f"Flatten size: {flat_image.size()}")

# example of nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f"Output size of hidden layer: {hidden1.size()}")

# example of ReLu
print(f"Before ReLu: {hidden1}\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLu: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(f"Output logits of sequential: {logits}")

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"Predicted probability of sequential: {pred_probab}")

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}\n")
