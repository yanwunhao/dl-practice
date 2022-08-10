import numpy as np
from torchvision import datasets, transforms


def mnist_iid(datasets, num_users):
    num_items = int(len(datasets) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(datasets))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
