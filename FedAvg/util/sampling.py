import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    # spilt samples into shards
    number_of_each_shard = 300
    num_shards = int(len(dataset) / number_of_each_shard)

    # create dictionary mapping samples to users
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # calculate valid samples index
    idxs = np.arange(num_shards * number_of_each_shard)
    labels = dataset.targets.numpy()[:len(idxs)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, int(len(idx_shard) / num_users), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*number_of_each_shard:(rand+1)*number_of_each_shard]), axis=0)
    return dict_users
