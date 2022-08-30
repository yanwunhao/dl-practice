import numpy as np
import torch
from torchvision import datasets, transforms

from model.neuralnetwork import CnnForMnist
from util.args import args_parser
from util.sampling import mnist_iid, mnist_noniid

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

print(dict_users)
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
for iter in range(args.epochs):
    loss_locals = []
    if not args.all_clients:
        w_locals = []
    # select valid users randomly
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    print("Epoch: ", iter, " Selected Users: ", idxs_users)
