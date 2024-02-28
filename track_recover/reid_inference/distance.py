import torch
import torch.nn.functional as F
from .re_ranking import re_ranking


def compute_distance_matrix(input1, input2, metric='euclidean'):
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'The 1st input need to be 2-D dimension, instead it is {}-D'.format(input1.dim())
    assert input2.dim() == 2, 'The 2nd input need to be 2-D dimension, instead it is {}-D'.format(input2.dim())
    assert input1.size(1) == input2.size(1), 'The 2 inputs does not have the same features space'

    if metric == 'manhattan':
        distmat = manhattan_distance(input1, input2)
    elif metric == 'euclidean':
        distmat = euclidean_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric. '
            'Please select between "manhattan", "euclidean", or "cosine".'
            )
    
    return distmat


def manhattan_distance(input1, input2):
    """
    Calculate the squared euclidean distance

    Return a distmat of size (n, m)
    """
    n, m, d = input1.size(0), input2.size(0), input1.size(1)

    input1 = input1.unsqueeze(1).expand(n, m, d)
    input2 = input2.unsqueeze(0).expand(n, m, d)

    distmat = torch.abs(input1 - input2).sum(dim=2)

    return distmat


def euclidean_distance(input1, input2):
    """
    Calculate the squared euclidean distance

    Return a distmat of size (n, m)
    """
    n, m, d = input1.size(0), input2.size(0), input1.size(1)

    input1 = input1.unsqueeze(1).expand(n, m, d)
    input2 = input2.unsqueeze(0).expand(n, m, d)

    distmat = torch.pow((input1 - input2), 2).sum(dim=2)

    return distmat


def cosine_distance(input1, input2):
    """
    Calculate the squared euclidean distance

    Return a distmat of size (n, m)
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)

    distmat = 1 - torch.mm(input1_normed, input2_normed.t())

    return distmat