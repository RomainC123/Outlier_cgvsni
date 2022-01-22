import numpy as np
import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from params.constants import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS
from NICE import NICE

################################################################################
#   VGG Model class                                                            #
################################################################################


class NcgNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, norm_layer=nn.BatchNorm2d, nonlinear='relu'):
        super(NcgNetBlock, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        conv_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=use_bias),
            norm_layer(out_channels)
        ]

        if nonlinear == 'elu':
            conv_block.append(nn.ELU(True))
        elif nonlinear == 'relu':
            conv_block.append(nn.ReLU(True))
        else:
            conv_block.append(nn.LeakyReLU(0.2, True))

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.model(x)


class HybridNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, norm_layer=nn.BatchNorm2d, nonlinear='relu', pooling='max', firstmost=False):
        super(HybridNetBlock, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if firstmost == True:
            use_bias = True

        conv_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=use_bias)
        ]

        if firstmost == False:
            conv_block.append(norm_layer(out_channels))

        if nonlinear == 'elu':
            conv_block.append(nn.ELU(True))
        elif nonlinear == 'relu':
            conv_block.append(nn.ReLU(True))
        elif nonlinear == 'leakyrelu':
            conv_block.append(nn.LeakyReLU(0.2, True))

        if pooling is not None:
            conv_block.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.model(x)


class ENet(nn.Module):
    def __init__(self):
        super(ENet, self).__init__()

        self.input_dim = INPUT_DIM

        self.convFilter0 = nn.Conv2d(3, 30, 5, bias=False)
        self.branch0 = nn.Sequential(HybridNetBlock(30, 64, nonlinear=None, pooling=None),
                                     HybridNetBlock(64, 64, pooling=None),
                                     HybridNetBlock(64, 64, pooling=None)
                                     )

        self.convFilter1_r = nn.Conv2d(1, 30, 5, bias=False)
        self.convFilter1_g = nn.Conv2d(1, 30, 5, bias=False)
        self.convFilter1_b = nn.Conv2d(1, 30, 5, bias=False)
        self.branch1 = nn.Sequential(HybridNetBlock(90, 64, nonlinear=None, pooling=None),
                                     HybridNetBlock(64, 64, pooling=None),
                                     HybridNetBlock(64, 64, pooling=None)
                                     )

        self.block1 = NcgNetBlock(128, 64, 7, stride=2, norm_layer=nn.BatchNorm2d)
        self.block2 = NcgNetBlock(64, 48, 5, norm_layer=nn.BatchNorm2d)
        self.block3 = NcgNetBlock(48, 64, 3, norm_layer=nn.BatchNorm2d)

        self.NICE = NICE(input_dim=self.input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)  # Maybe need to find input_dim here

    def forward(self, input):
        x = self.convFilter0(input)
        x = self.branch0(x)

        y = torch.chunk(input, 3, dim=1)
        y_0 = self.convFilter1_r(y[0])
        y_1 = self.convFilter1_g(y[1])
        y_2 = self.convFilter1_b(y[2])
        y_cat = torch.cat((y_0, y_1, y_2), 1)
        y = self.branch1(y_cat)

        x_y = torch.cat((x, y), dim=1)

        x = self.block1(x_y)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.block2(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.block3(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = x.view(x.size(0), -1)
        # Flow model starting on that feature map
        x = self.NICE(x)
        return x

    def init_convFilter(self, trainable=False):
        srm = np.loadtxt('SRM1505.txt')
        srm55 = np.reshape(srm, (30, 5, 5))
        srm55 = torch.Tensor(srm55)
        srm55 = srm55.unsqueeze(1)
        self.convFilter1_r.weight.data[...] = srm55[...]
        self.convFilter1_g.weight.data[...] = srm55[...]
        self.convFilter1_b.weight.data[...] = srm55[...]

        if not trainable:
            self.convFilter1_r.weight.requires_grad = False
            self.convFilter1_g.weight.requires_grad = False
            self.convFilter1_b.weight.requires_grad = False

################################################################################
#   VGG Model container                                                        #
################################################################################


class ENetWrapper:

    def __init__(self):
        self.model = ENet()

    def _get_nb_parameters(self):

        nb_params = 0
        for param in self.model.parameters():
            nb_params += param.numel()

        return nb_params

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict()},
                   os.path.join(path, f'checkpoint_{epoch}.pth'))

    def get_info(self):

        infos = str(self.model)
        infos += f'\nNumber of parameters: {self._get_nb_parameters()}'

        return infos