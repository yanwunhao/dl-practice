import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader


def model_evaluation(net, dataset, args):
    net.eval()

    test_loss = 0
    correct = 0

    data_loader = DataLoader(dataset, batch_size=args.eval_bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.device == 'gpu':
            data, target = data.cuda(), target.cuda()
        log_probs = net(data)
        # summarize batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction="sum")
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    return accuracy, test_loss
