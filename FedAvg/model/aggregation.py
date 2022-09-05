import copy
import torch


def FedAvg(w):
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]
        weights_avg[k] = torch.div(weights_avg[k], len(w))
    return weights_avg
