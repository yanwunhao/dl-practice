import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    # other arguments
    parser.add_argument('--iid', action='store_true', default=True, help='whether i.i.d or not')

    args = parser.parse_args()
    return args
