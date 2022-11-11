import torch
import os
import random
import numpy as np
import yaml
from collections import defaultdict

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


def print_and_save_args(args, path):
    # let's save the args as json to enable easy loading
    with open(os.path.join(path, 'config.yml'), 'w') as f:
        yaml.dump(args, f)


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return

        str_tp = str(type(value[0]))
        if type(value[0]) == torch.Tensor:
            value = torch.mean(torch.stack(value))
        elif 'float' in str_tp or 'int' in str_tp:
            value = sum(value) / len(value)
    zeros = 40 - len(name)
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)


def reset_log():
    return defaultdict(list)
    logs = OD()
    for name in ['inner log p(x|z)', 'log p(x|z)', 'log p(x|z) nn', 'commit', 'vq', 'kl', 'bpd', 'elbo']:
        logs[name] = []
    return logs


def scale_inv(images):
    return images + 0.5