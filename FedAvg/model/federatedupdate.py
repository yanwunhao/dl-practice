import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class LocalDataset(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idx=None):
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(LocalDataset(dataset, idx), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()

        # Train and Update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_fn(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)