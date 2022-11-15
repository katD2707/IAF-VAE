import torch


def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    mask = torch.ones([n_in, n_out, h, w], dtype=torch.float32)
    mask[:, :, h // 2, w // 2 + zerodiagonal:] = 0
    mask[:, :, h // 2 + 1:] = 0

    return mask
