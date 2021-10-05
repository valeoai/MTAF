import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target < 200)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def kl_divergence(predict_0, predict_1):
    """
    Args:
        predict_0:(n, c, h, w)
        predict_1:(n, c, h, w)
    """
    assert predict_0.dim() == 4
    assert predict_1.dim() == 4
    assert predict_0.size(0) == predict_1.size(0), f"{predict_0.size(0)} vs {predict_1.size(0)}"
    assert predict_0.size(1) == predict_1.size(1), f"{predict_0.size(1)} vs {predict_1.size(1)}"
    assert predict_0.size(2) == predict_1.size(2), f"{predict_0.size(2)} vs {predict_1.size(2)}"
    assert predict_0.size(3) == predict_1.size(3), f"{predict_0.size(3)} vs {predict_1.size(3)}"
    n, c, h, w = predict_0.size()
    predict_0 = predict_0.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    predict_1 = predict_1.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    softmax_predict_0 = F.softmax(predict_0)
    softmax_predict_1 = F.softmax(predict_1)
    log_softmax_predict_0 = F.log_softmax(predict_0)
    loss = F.kl_div(log_softmax_predict_0,softmax_predict_1,size_average=True)
    return loss

def mse_loss(predict_0, predict_1):
    """
    Args:
        predict_0:(n, c, h, w)
        predict_1:(n, c, h, w)
    """
    assert predict_0.dim() == 4
    assert predict_1.dim() == 4
    assert predict_0.size(0) == predict_1.size(0), f"{predict_0.size(0)} vs {predict_1.size(0)}"
    assert predict_0.size(1) == predict_1.size(1), f"{predict_0.size(1)} vs {predict_1.size(1)}"
    assert predict_0.size(2) == predict_1.size(2), f"{predict_0.size(2)} vs {predict_1.size(2)}"
    assert predict_0.size(3) == predict_1.size(3), f"{predict_0.size(3)} vs {predict_1.size(3)}"
    n, c, h, w = predict_0.size()
    predict_0 = predict_0.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    predict_1 = predict_1.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    softmax_predict_0 = F.softmax(predict_0)
    softmax_predict_1 = F.softmax(predict_1)
    softmax_mask = ((torch.max(softmax_predict_0, dim=1, keepdim=True)[0].expand(-1, c)) > threshold)
    softmax_predict_0 = softmax_predict_0[softmax_mask]
    softmax_predict_1 = softmax_predict_1[softmax_mask]
    loss = F.mse_loss(softmax_predict_0,softmax_predict_1,size_average=True)
    return loss
