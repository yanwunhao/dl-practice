import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_users', type=int, default=50, help="number of users: K")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    # other arguments
    parser.add_argument('--iid', action='store_true', default=False, help='whether i.i.d or not')
    parser.add_argument('--all_clients', action='store_true', default=True, help='aggregation over all clients')

    args = parser.parse_args()
    return args
