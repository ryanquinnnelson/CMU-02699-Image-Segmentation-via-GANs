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
        return self.blocks(x)


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
        return self.linear_block(x)


class SegmentationNetwork(nn.Module):
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

        self.block4 = nn.Sequential(
            CnnBlock(2048, 1024),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=1, bias=False),
            nn.Softmax(dim=1)  # ?? dim1, number of classes
        )

    def forward(self, x):
        block1out = self.block1(x)
        block2out = self.block2(block1out)
        block3out = self.block3(block2out)

        # concatenate results
        concatenated = torch.cat((block1out, block2out, block3out))

        block4out = self.block4(concatenated)

        return block4out


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

            LinearBlock(512, 256),
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            nn.Linear(64, 1),  # or 2?
            nn.Sigmoid()

        )

    def forward(self, x):
        block1out = self.block1(x)
        block2out = self.block2(block1out)

        return block2out
