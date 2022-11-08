import torch
import torch.nn as nn
import torch.nn.functional as F
from masks import get_conv_ar_mask
from utils import *

class CVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 z_size,
                 batch_size,
                 k,
                 kl_min,
                 num_hidden_layers,
                 num_blocks,
                 image_size,
                 mode='train',
                 *args,
                 **kwargs):
        super(CVAE, self).__init__()
        self.h_size = hidden_size
        self.z_size = z_size
        self.mode = mode
        self.data_size = batch_size * k
        self.image_size = image_size
        self.k = k

        # Encoder input     # (B, hidden_size, 16, 16)
        self.x_enc = nn.Conv2d(in_channels,
                               self.h_size,
                               kernel_size=(5, 5),
                               stride=2,
                               padding=2,
                               )

        # Inference and generative
        self.layers = []
        for i in range(num_hidden_layers):
            self.layers.append([])
            for j in range(num_blocks):
                downsample = (i > 0) and (j == 0)
                self.layers[-1].append(IAFLayer(in_channels,
                                                z_size,
                                                hidden_size,
                                                num_hidden_layers,
                                                batch_size,
                                                kl_min,
                                                mode,
                                                k,
                                                downsample)
                                       )
                in_channels = self.h_size

        self.activation = nn.ELU()

        # Decoder output
        self.x_dec = nn.ConvTranspose2d(in_channels=self.h_size,
                                        out_channels=3,
                                        kernel_size=5,
                                        stride=2,
                                        padding=2,
                                        output_padding=1,
                                        )

    def forward(self, inputs):
        inputs = torch.tile(inputs, [2, 1, 1, 1])

        h = self.x_enc(inputs)
        # Bottom up
        for layer in self.layers:
            for sub_layer in layer:
                h = sub_layer.up(h)

        h = torch.zeros((self.data_size,
                         self.h_size,
                         self.image_size // 2 ** len(self.layers),
                         self.image_size // 2 ** len(self.layers)
                         ))
        kl_cost = kl_obj = 0.

        # Top down
        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, cur_obj, cur_cost = sub_layer.down(h)
                kl_obj += cur_obj
                kl_cost += cur_cost

                # tensorboard implementation
                # if self.mode == "train":

        h = self.activation(h)
        h = self.x_dec(h)
        h = torch.clamp(h, -0.5 + 1 / 512., 0.5 - 1 / 512.)

        log_pxz = discretized_logistic(h, self.dec_log_stdv, sample=inputs)
        obj = (kl_obj - log_pxz).sum()

        # tensorboard implementation
        # if self.mode == 'train':

        loss = compute_lowerbound(log_pxz, kl_cost, self.k).sum()

        return h, obj, loss


class IAFLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 z_size,
                 h_size,
                 num_hidden_layers,
                 batch_size,
                 kl_min,
                 mode,
                 k=1,
                 downsample=False,
                 *args,
                 **kwargs,
                 ):
        super(IAFLayer, self).__init__()
        self.mode = mode
        self.down_sample = downsample
        self.z_size = z_size
        self.h_size = h_size
        self.num_hidden_layers = num_hidden_layers
        self.k = k
        self.batch_size = batch_size
        self.kl_min = kl_min

        # Inference model
        # Bottom up mean, standard deviation and context
        self.mean_qz, self.log_std_qz, self.up_context = None, None, None  # need to store for bidirectional inference

        self.activation = nn.ELU()
        self.conv2d_up_1 = nn.Conv2d(in_channels,
                                     out_channels=2 * self.z_size + 2 * self.h_size,
                                     stride=1,
                                     kernel_size=3,
                                     padding=1,
                                     )
        self.conv2d_up_2 = nn.Conv2d(in_channels=self.h_size,
                                     out_channels=self.h_size,
                                     stride=1,
                                     kernel_size=3,
                                     padding=1,
                                     )

        # Generative model
        self.conv2d_down_1 = nn.Conv2d(in_channels=in_channels,
                                       out_channels=4 * self.z_size + 2 * self.h_size,
                                       stride=1,
                                       kernel_size=3,
                                       padding=1,
                                       )
        self.multi_masked_conv2d = AutoregressiveMultiConv2d(in_channels=self.z_size,
                                                             hidden_layers=self.num_hidden_layers * [self.h_size],
                                                             out_channels=2 * self.z_size)
        if downsample:
            self.deconv2d = nn.ConvTranspose2d(in_channels=self.h_size + self.z_size,
                                               out_channels=self.h_size,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1,
                                               )
        else:
            self.conv2d_down_2 = nn.Conv2d(self.h_size + self.z_size,
                                           self.h_size,
                                           kernel_size=3,
                                           padding=1,
                                           )

    def up(self, inputs):
        h = self.activation(inputs)
        h = self.conv2d_up_1(h)
        self.mean_qz, self.log_std_qz, self.up_context, h = torch.tensor_split(h, (
            self.z_size, 2 * self.z_size, 2 * self.z_size + self.h_size), dim=1)
        h = self.activation(h)
        h = self.conv2d_up_2(h)
        if self.down_sample:
            inputs = F.interpolate(inputs, scale_factor=(0.5, 0.5), mode='nearest')
        return inputs + 0.1 * h

    def down(self, inputs):
        x = self.activation(inputs)
        x = self.conv2d_down_1(x)
        mean_pz, log_std_pz, mean_rz, log_std_rz, down_context, h_det = torch.tensor_split(x, (self.z_size,
                                                                                               2 * self.z_size,
                                                                                               3 * self.z_size,
                                                                                               4 * self.z_size,
                                                                                               4 * self.z_size + self.h_size),
                                                                                           dim=1)
        eps_prior = torch.randn(mean_pz.shape[0], self.z_size, mean_pz.shape[-2], mean_pz.shape[-1])
        eps_posterior = torch.randn(mean_rz.shape[0], self.z_size, mean_rz.shape[-2], mean_rz.shape[-1])
        prior = mean_pz + eps_prior * torch.exp(log_std_pz)
        mean_posterior = mean_rz + self.mean_qz
        log_std_posterior = self.log_std_qz + log_std_rz
        posterior = mean_posterior + eps_posterior * log_std_posterior
        context = down_context + self.up_context

        if self.mode in ['init', 'sample']:
            z = prior
        else:
            z = posterior

        if self.mode == 'sample':
            kl_cost = kl_obj = torch.zeros((self.batch_size * self.k))
        else:
            log_qz = -0.5 * (torch.log(2 * np.pi * torch.pow(torch.exp(log_std_posterior), 2)) + torch.pow(
                z - mean_posterior, 2) / torch.exp(log_std_posterior))  # posterior loss
            arw_mean, arw_log_std = self.multi_masked_conv2d(z, context)
            z = (z - arw_mean) / torch.exp(arw_log_std)
            log_qz += arw_log_std
            log_pz = -0.5 * (torch.log(2 * np.pi * torch.pow(torch.exp(log_std_pz), 2)) + torch.pow(z - mean_pz,
                                                                                                    2) / torch.exp(
                log_std_pz))

            kl_cost = log_qz - log_pz

            if self.kl_min > 0:
                kl_obj = kl_cost.sum(dim=(2, 3)).mean(0, keepdim=True)
                kl_obj = torch.maximum(kl_obj, self.kl_min)
                kl_obj = kl_obj.sum()
                kl_obj = torch.tile(kl_obj, [self.batch_size * self.k])
            else:
                kl_obj = kl_cost.sum(dim=(1, 2, 3))

            kl_cost = kl_cost.sum(dim=(1,2,3))

        h = torch.cat((z, h_det), dim=1)
        h = self.activation(h)
        if self.down_sample:
            inputs = F.interpolate(inputs, scale_factor=(2, 2), mode='nearest')
            h = self.deconv2d(h)  #
        else:
            h = self.conv2d_down_2(h)
        output = inputs + 0.1 * h
        return output, kl_obj, kl_cost  # (h_size, ..., ...), (B, ), (z_size, ..., ...)


class AutoregressiveMultiConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_layers,
                 ):
        super(AutoregressiveMultiConv2d, self).__init__()

        self.MaskedLayers = nn.Sequential(*[
            MaskedConv2d(in_channel, out_channel, diag_mask=False)
            for in_channel, out_channel in zip([in_channels] + hidden_layers[:-1], hidden_layers)
        ])

        self.MaskedConv2dMean = MaskedConv2d(in_channels=hidden_layers[-1],
                                             out_channels=out_channels,
                                             diag_mask=True,
                                             )
        self.MaskedConv2dStd = MaskedConv2d(in_channels=hidden_layers[-1],
                                            out_channels=out_channels,
                                            diag_mask=True,
                                            )

    def forward(self, inputs, context):
        x = self.MaskedLayers(inputs, context)
        mean = self.MaskedConv2dMean(x)
        std = self.MaskedConv2dStd(x)
        return mean, std


class MaskedConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 kernel_size=3,
                 padding=1,
                 diag_mask=False,
                 ):
        super(MaskedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        mask = get_conv_ar_mask(kernel_size, kernel_size, in_channels, out_channels, zerodiagonal=diag_mask)
        mask = mask.reshape(out_channels, in_channels, kernel_size, kernel_size)
        self.register_buffer('mask', mask)
        self.elu = nn.ELU()

    def forward(self, inputs, context):
        return self.elu(context + F.conv2d(inputs, self.mask * self.conv2d.weight, self.conv2d.bias))
