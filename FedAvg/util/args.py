import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--frac', type=float, default=0.3, help="the fraction of clients: C")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_users', type=int, default=50, help="number of users: K")

    # federated client arguments
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    # other arguments
    parser.add_argument('--iid', action='store_true', default=False, help='whether i.i.d or not')
    parser.add_argument('--all_clients', action='store_true', default=False, help='aggregation over all clients')
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose print')

    args = parser.parse_args()
    return args
