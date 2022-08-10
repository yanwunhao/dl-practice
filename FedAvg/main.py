import torch
from torchvision import datasets, transforms

from model.neuralnetwork import CnnForMnist
from util.args import args_parser
from util.sampling import mnist_iid

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

print(args)

# load dataset and split users
if args.dataset == "mnist":
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_for_train = datasets.MNIST('../data/', train=True, download=True, transform=trans_mnist)
    dataset_for_test = datasets.MNIST('../data/', train=False, download=True, transform=trans_mnist)

    if args.iid:
        dict_users = mnist_iid(dataset_for_train, args.num_users)
    else:
        pass

else:
    exit("ERROR: No Recognized Dataset")

img_size = dataset_for_train[0][0].shape

# build model
if args.model == "cnn":
    net_glob = CnnForMnist(args).to(args.device)

print(net_glob)
net_glob.train()
