import numpy as np
import torch
from torchvision import datasets, transforms

from model.neuralnetwork import CnnForMnist
from model.federatedupdate import LocalUpdate
from util.args import args_parser
from util.sampling import mnist_iid, mnist_noniid

import copy

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

# print(args)

# load dataset and split users
if args.dataset == "mnist":
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_for_train = datasets.MNIST('../data/', train=True, download=True, transform=trans_mnist)
    dataset_for_test = datasets.MNIST('../data/', train=False, download=True, transform=trans_mnist)

    if args.iid:
        dict_users = mnist_iid(dataset_for_train, args.num_users)
    else:
        dict_users = mnist_noniid(dataset_for_train, args.num_users)

else:
    exit("ERROR: NO RECOGNIZED DATASET")

img_size = dataset_for_train[0][0].shape

# build model
if args.model == "cnn":
    net_glob = CnnForMnist(args).to(args.device)

# print(net_glob)
net_glob.train()

# copy weights
w_glob = net_glob.state_dict()

# training
loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []

if args.all_clients:
    print("Aggregation over all clients")
    w_locals = [w_glob for i in range(args.num_users)]
else:
    w_locals = []
for iter in range(args.epochs):
    loss_locals = []
    # select valid users randomly
    m = max(int(args.frac * args.num_users), 1)
    selected_users = np.random.choice(range(args.num_users), m, replace=False)
    print("Epoch: ", iter, " Selected Users: ", selected_users)
    for idx in selected_users:
        local = LocalUpdate(args=args, dataset=dataset_for_train, idx=dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        if args.all_clients:
            w_locals[idx] = copy.deepcopy(w)
        else:
            w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
    # update glob weights
