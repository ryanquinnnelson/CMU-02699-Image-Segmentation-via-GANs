"""
Everything related to Generative Adversarial Networks.
"""
__author__ = 'ryanquinnnelson'

import logging
from collections import OrderedDict

import torch
import torch.nn as nn


def _calc_output_size(input_size, padding, dilation, kernel_size, stride):
    """
    Calculate the output size based on all parameters.
    Args:
        input_size (int): size of the input
        padding (int): amount of padding
        dilation (int): amount of dilation
        kernel_size (int): size of the kernel
        stride (int): size of the stride
    Returns: int representing output size
    """
    input_size_padded = input_size + 2 * padding
    kernel_dilated = (kernel_size - 1) * (dilation - 1) + kernel_size
    output_size = (input_size_padded - kernel_dilated) // stride + 1
    return output_size


def _init_weights(layer):
    """
    Perform initialization of layer weights if layer is a Conv2d layer.
    Args:
        layer: layer under consideration
    Returns: None
    """
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)


class CnnBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # initialize weights
        self.cnn_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'cnn_block_input:{x.shape}')
        x = self.cnn_block(x)
        # logging.info(f'cnn_block:{x.shape}')
        return x


# bi-linear interpolation, or learned up-sampling filters

# nn.Upsample(size=None, scale_factor=None, mode='bilinear')


# nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear')
# https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, output_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_block = nn.Sequential(
            nn.Upsample(size=output_size, mode='bilinear'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # initialize weights
        self.up_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'up_block_input:{x.shape}')
        x = self.up_block(x)
        # logging.info(f'up_block:{x.shape}')
        return x


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear_block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )

        self.linear_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'linear_block_input:{x.shape}')
        x = self.linear_block(x)
        # logging.info(f'linear_block:{x.shape}')
        return x


class SegmentationNetwork(nn.Module):
    def __init__(self, in_features, input_size):
        super().__init__()

        self.block1 = nn.Sequential(
            # conv1
            CnnBlock(in_features, 64),
            CnnBlock(64, 64),

            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv2
            CnnBlock(64, 128),

            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv3
            CnnBlock(128, 128),
            CnnBlock(128, 256),

            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv4
            CnnBlock(256, 512),
            CnnBlock(512, 512)  # shortcut to up-conv1

        )

        self.block2 = nn.Sequential(

            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv5
            CnnBlock(512, 512),
            CnnBlock(512, 512)  # shortcut to up-conv2

        )

        self.block3 = nn.Sequential(

            # pool5
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv6
            CnnBlock(512, 1024),
            CnnBlock(1024, 1024)  # shortcut to up-conv3
        )

        self.block4 = UpConvBlock(1024, 1024, (224, 332))

        self.block5 = UpConvBlock(512, 512, (224, 332))

        self.block6 = UpConvBlock(512, 512, (224, 332))

        self.block7 = nn.Sequential(
            CnnBlock(2048, 1024),
            nn.Conv2d(1024, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):
        block1out = self.block1(x)
        block2out = self.block2(block1out)
        block3out = self.block3(block2out)

        # upconvolution
        block4out = self.block4(block3out)
        block5out = self.block5(block2out)
        block6out = self.block6(block1out)

        # concatenate results
        concatenated = torch.cat((block4out, block5out, block6out), dim=1)  # channels are the second dimension

        block7out = self.block7(concatenated)

        if i == 0:
            logging.info(f'block1out.shape:{block1out.shape}')
            logging.info(f'block2out.shape:{block2out.shape}')
            logging.info(f'block3out.shape:{block3out.shape}')
            logging.info(f'block4out.shape:{block4out.shape}')
            logging.info(f'block5out.shape:{block5out.shape}')
            logging.info(f'block6out.shape:{block6out.shape}')
            logging.info(f'concatenated.shape:{concatenated.shape}')
            logging.info(f'block7out.shape:{block7out.shape}')

        return block7out


class EvaluationNetwork(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block1 = nn.Sequential(

            # conv1
            CnnBlock(in_features, 64),
            CnnBlock(64, 64),

            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv2
            CnnBlock(64, 128),

            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv3
            CnnBlock(128, 256),
            CnnBlock(256, 256),

            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv4
            CnnBlock(256, 512),
            CnnBlock(512, 512),

            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv5
            CnnBlock(512, 512),
            CnnBlock(512, 512)

        )

        self.block2 = nn.Sequential(

            nn.Flatten(),  # need to convert 2d to 1d


        )

        self.block3 = nn.Sequential(
            LinearBlock(168960, 256),  # 512*15*22
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            nn.Linear(64, 1),  # binary classes
            nn.Sigmoid()
        )

    def forward(self, x, i):
        if i == 0:
            logging.info(f'x:{x.shape}')

        block1out = self.block1(x)

        if i == 0:
            logging.info(f'block1out:{block1out.shape}')
        block2out = self.block2(block1out)

        if i == 0:
            logging.info(f'block2out:{block2out.shape}')

        block3out = self.block3(block2out)

        if i == 0:
            logging.info(f'block3out:{block3out.shape}')

        return block3out
