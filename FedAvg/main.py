import numpy as np
import torch
from torchvision import datasets, transforms

from util.args import args_parser
from util.sampling import mnist_iid, mnist_noniid
from model.neuralnetwork import CnnForMnist
from model.federatedupdate import LocalUpdate
from model.aggregation import FedAvg
from model.test import model_evaluation

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
    w_glob = FedAvg(w_locals)

    # copy weights to global model
    net_glob.load_state_dict(w_glob)

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)

# evaluation
net_glob.eval()
acc_train, loss_train = model_evaluation(net_glob, dataset_for_train, args)
acc_test, loss_test = model_evaluation(net_glob, dataset_for_test, args)
print("Training Accuracy: {:.2f}".format(acc_train))
print("Test Accuracy: {:.2f}".format(acc_test))
