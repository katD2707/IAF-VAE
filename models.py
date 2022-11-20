import torch
import torch.nn as nn
import torch.nn.functional as F
from masks import get_conv_ar_mask
from utils import *
from torchsummary import summary


class CVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 z_size,
                 k,
                 kl_min,
                 num_hidden_layers,
                 num_blocks,
                 image_size,
                 device='cpu',
                 *args,
                 **kwargs):
        super(CVAE, self).__init__()
        self.h_size = hidden_size
        self.z_size = z_size
        self.image_size = image_size
        self.k = k
        self.device = device

        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))

        # Encoder input     # (B, hidden_size, 16, 16)
        self.x_enc = nn.Conv2d(in_channels,
                               self.h_size,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1,
                               )

        # Inference and generative
        layers = []
        for i in range(num_hidden_layers):
            layer = []
            for j in range(num_blocks):
                downsample = (i > 0) and (j == 0)
                layer.append(IAFLayer(hidden_size,
                                      z_size,
                                      hidden_size,
                                      num_hidden_layers,
                                      kl_min,
                                      k,
                                      downsample)
                             )
            layers.append(nn.ModuleList(layer))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU()

        # Decoder output
        self.x_dec = nn.ConvTranspose2d(in_channels=self.h_size,
                                        out_channels=in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        )

    def forward(self, inputs):
        h = self.x_enc(inputs)

        # Bottom up
        for layer in self.layers:
            for sub_layer in layer:
                h = sub_layer.up(h)

        h = torch.zeros((inputs.shape[0],
                         self.h_size,
                         self.image_size // 2 ** len(self.layers),
                         self.image_size // 2 ** len(self.layers)
                         )).to(inputs.device)
        kl = kl_obj = 0.

        # Top down
        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, cur_obj, cur_cost = sub_layer.down(h)
                kl_obj += cur_obj
                kl += cur_cost

        h = self.activation(h)
        h = self.x_dec(h)
        h = torch.clamp(h, -0.5 + 1 / 512., 0.5 - 1 / 512.)

        log_pxz = discretized_logistic(h, self.dec_log_stdv, sample=inputs)
        obj = (kl_obj - log_pxz).sum()

        elbo = compute_lowerbound(log_pxz, kl, self.k)

        return h, obj, elbo

    def sample(self, n_samples=64):
        h = torch.zeros((n_samples * self.k,
                         self.h_size,
                         self.image_size // 2 ** len(self.layers),
                         self.image_size // 2 ** len(self.layers),
                         )).to(self.device)

        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, _, _ = sub_layer.down(h, mode='sample')

        h = F.elu(h)
        h = self.x_dec(h)

        return h.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

    def cond_sample(self, inputs):
        h = self.x_enc(inputs)

        for layer in self.layers:
            for sub_layer in layer:
                h = sub_layer.up(h)

        h = torch.zeros((inputs.size(0),
                         self.h_size,
                         self.image_size // 2 ** len(self.layers),
                         self.image_size // 2 ** len(self.layers),
                         )).to(self.device)

        outs = []

        current = 0
        for i, layer in enumerate(reversed(self.layers)):
            for j, sub_layer in enumerate(reversed(layer)):
                h, _, _ = sub_layer.down(h)

                h_copy = h
                again = 0
                # now, sample the rest of the way:
                for layer_ in reversed(self.layers):
                    for sub_layer_ in reversed(layer_):
                        if again > current:
                            h_copy, _, _ = sub_layer_.down(h_copy, mode='sample')

                        again += 1

                x = F.elu(h_copy)
                x = self.x_dec(x)
                x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
                outs += [x]

                current += 1

        return outs


class IAFLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 z_size,
                 h_size,
                 num_hidden_layers,
                 kl_min,
                 k=1,
                 downsample=False,
                 *args,
                 **kwargs,
                 ):
        super(IAFLayer, self).__init__()

        self.down_sample = downsample
        self.z_size = z_size
        self.h_size = h_size
        self.num_hidden_layers = num_hidden_layers
        self.k = k
        self.kl_min = kl_min

        # Inference model
        # Bottom up mean, standard deviation and context
        self.mean_qz, self.log_std_qz, self.up_context = None, None, None  # need to store for bidirectional inference

        self.activation = nn.ELU()

        if downsample:
            down_stride, kernel_size = 2, 4
        else:
            down_stride, kernel_size = 1, 3

        self.conv2d_up_1 = nn.utils.weight_norm(nn.Conv2d(in_channels,
                                                          out_channels=2 * self.z_size + 2 * self.h_size,
                                                          stride=down_stride,
                                                          kernel_size=kernel_size,
                                                          padding=1,
                                                          )
                                                )

        self.conv2d_up_2 = nn.utils.weight_norm(nn.Conv2d(in_channels=self.h_size,
                                                          out_channels=self.h_size,
                                                          stride=1,
                                                          kernel_size=3,
                                                          padding=1,
                                                          )
                                                )

        # Generative model
        self.conv2d_down_1 = nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels,
                                                            out_channels=4 * self.z_size + 2 * self.h_size,
                                                            stride=1,
                                                            kernel_size=3,
                                                            padding=1,
                                                            )
                                                  )
        self.multi_masked_conv2d = AutoregressiveMultiConv2d(in_channels=self.z_size,
                                                             hidden_layers=self.num_hidden_layers * [self.h_size],
                                                             out_channels=self.z_size)
        if downsample:
            self.deconv2d = nn.utils.weight_norm(nn.ConvTranspose2d(in_channels=self.h_size + self.z_size,
                                                                    out_channels=self.h_size,
                                                                    kernel_size=4,
                                                                    stride=2,
                                                                    padding=1,
                                                                    )
                                                 )
        else:
            self.conv2d_down_2 = nn.utils.weight_norm(nn.Conv2d(self.h_size + self.z_size,
                                                                self.h_size,
                                                                kernel_size=3,
                                                                padding=1,
                                                                )
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

    def down(self, inputs, mode='train'):
        data_size = inputs.shape[0] * self.k
        x = self.activation(inputs)
        x = self.conv2d_down_1(x)
        mean_pz, log_std_pz, mean_rz, log_std_rz, down_context, h_det = torch.tensor_split(x, (self.z_size,
                                                                                               2 * self.z_size,
                                                                                               3 * self.z_size,
                                                                                               4 * self.z_size,
                                                                                               4 * self.z_size + self.h_size),
                                                                                           dim=1)
        eps_prior = torch.randn(data_size, self.z_size, mean_pz.shape[-2], mean_pz.shape[-1]).to(inputs.device)
        eps_posterior = torch.randn(data_size, self.z_size, mean_rz.shape[-2], mean_rz.shape[-1]).to(inputs.device)
        prior = mean_pz + eps_prior * torch.exp(log_std_pz)
        mean_posterior = mean_rz + self.mean_qz
        log_std_posterior = self.log_std_qz + log_std_rz
        posterior = mean_posterior + eps_posterior * torch.exp(log_std_posterior)
        context = down_context + self.up_context

        if mode in ['init', 'sample']:
            z = prior
            kl_cost = kl_obj = torch.zeros(data_size).to(inputs.device)
        else:
            z = posterior

            log_qz = -0.5 * (torch.log(2 * np.pi * torch.pow(torch.exp(log_std_posterior), 2)) + torch.pow(
                z - mean_posterior, 2) / torch.pow(torch.exp(log_std_posterior), 2))  # posterior loss

            arw_mean, arw_log_std = self.multi_masked_conv2d(z, context)
            arw_mean, arw_log_std = arw_mean * 0.1, arw_log_std * 0.1
            z = (z - arw_mean) / torch.exp(arw_log_std)

            log_qz += arw_log_std
            log_pz = -0.5 * (torch.log(2 * np.pi * torch.pow(torch.exp(log_std_pz), 2)) +
                             torch.pow(z - mean_pz, 2) / torch.pow(torch.exp(log_std_pz), 2))

            kl_cost = log_qz - log_pz

            if self.kl_min > 0:
                kl_obj = kl_cost.sum(dim=(2, 3)).mean(0, keepdim=True)
                kl_obj = kl_obj.clamp(min=self.kl_min)
                kl_obj = kl_obj.sum()
                kl_obj = torch.tile(kl_obj, [data_size])
            else:
                kl_obj = kl_cost.sum(dim=(1, 2, 3))

            kl_cost = kl_cost.sum(dim=(1, 2, 3))

        h = torch.cat((z, h_det), dim=1)
        h = self.activation(h)
        if self.down_sample:
            inputs = F.interpolate(inputs, scale_factor=(2, 2), mode='nearest')
            h = self.deconv2d(h)  #
        else:
            h = self.conv2d_down_2(h)

        return inputs + 0.1 * h, kl_obj, kl_cost


class AutoregressiveMultiConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_layers,
                 ):
        super(AutoregressiveMultiConv2d, self).__init__()

        self.MaskedLayers = nn.ModuleList([
            MaskedConv2d(diag_mask=False,
                         in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=3,
                         padding=1,
                         stride=1,
                         )
            for in_channel, out_channel in zip([in_channels] + hidden_layers[:-1], hidden_layers)
        ])

        self.MaskedConv2dMean = MaskedConv2d(diag_mask=True,
                                             in_channels=hidden_layers[-1],
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             padding=1,
                                             stride=1,
                                             )
        self.MaskedConv2dStd = MaskedConv2d(diag_mask=True,
                                            in_channels=hidden_layers[-1],
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            padding=1,
                                            stride=1,
                                            )

    def forward(self, inputs, context):
        for layer in self.MaskedLayers:
            inputs, context = layer(inputs, context)
        mean, _ = self.MaskedConv2dMean(inputs)
        std, _ = self.MaskedConv2dStd(inputs)

        return mean, std


class MaskedConv2d(nn.Conv2d):
    def __init__(self,
                 diag_mask=False,
                 *args,
                 **kwargs,
                 ):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        mask = get_conv_ar_mask(self.kernel_size[0], self.kernel_size[0], self.in_channels, self.out_channels,
                                zerodiagonal=diag_mask)
        mask = mask.reshape(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[0])
        self.register_buffer('mask', mask)
        self.elu = nn.ELU()

    def forward(self, inputs, context=None):
        inputs = self._conv_forward(inputs, self.mask * self.weight, self.bias)
        if context is not None:
            inputs += context
        return self.elu(inputs), context

# model = CVAE(
#     in_channels=3,
#     hidden_size=160,
#     z_size=32,
#     batch_size=64,
#     k=1,
#     kl_min=0.25,
#     num_hidden_layers=2,
#     num_blocks=2,
#     image_size=32,
#     mode='train',
# )
# summary(model, (3, 32, 32))
