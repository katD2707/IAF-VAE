import torch
import os
import random
import numpy as np


class Struct:
    """
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    """

    def __init__(self, **entries):
        self.entries = entries
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        """
        Return the only key in the Struct s.t. its value is True
        """
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def get_true_keys(self):
        """
        Return all the keys in the Struct s.t. its value is True
        """
        return [k for k, v in self.__dict__.items() if v == True]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    logp = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
    return logp.sum(dim=[1, 2, 3])


def compute_lowerbound(log_pxz: torch.Tensor, sum_kl_costs: torch.Tensor, k=1):
    if k == 1:
        return sum_kl_costs - log_pxz

    # log 1/k \sum p(x | z) * p(z) / q(z | x) = -log(k) + logsumexp(log p(x|z) + log p(z) - log q(z|x))
    log_pxz = log_pxz.reshape([-1, k])
    sum_kl_costs = sum_kl_costs.reshape([-1, k])
    return - (- torch.log(torch.Tensor(k, dtype=torch.float32)) + logsumexp(log_pxz - sum_kl_costs))


def logsumexp(x: torch.Tensor):
    x_max = x.max(dim=[1], keepdim=True)
    return x_max.reshape([-1]) + torch.log(torch.exp(x - x_max).sum([1]))


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')