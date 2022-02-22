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

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # self.input_size = input_size
        # self.output_size = _calc_output_size(input_size, padding, dilation, kernel_size, stride)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
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
# nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear')
# https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, size, mode='bilinear'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_block = nn.Sequential(
            nn.Upsample(size=size, mode=mode),
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

    def __init__(self, in_features, out_features, batchnorm=True, activation='relu'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear_block = nn.Sequential()
        self.linear_block.add_module('linear', nn.Linear(in_features, out_features))

        if batchnorm:
            self.linear_block.add_module('bn', nn.BatchNorm1d(out_features))

        if activation == 'relu':
            self.linear_block.add_module('activation', nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            self.linear_block.add_module('activation', nn.Sigmoid())

        self.linear_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'linear_block_input:{x.shape}')
        x = self.linear_block(x)
        # logging.info(f'linear_block:{x.shape}')
        return x


#
# class SegmentationNetwork(nn.Module):
#     def __init__(self, in_features, input_size):
#         super().__init__()
#
#         self.block1 = nn.Sequential(
#             # conv1
#             CnnBlock(in_features, 64),
#             CnnBlock(64, 64),
#
#             # pool1
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#
#             # conv2
#             CnnBlock(64, 128),
#
#             # pool2
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#
#             # conv3
#             CnnBlock(128, 128),
#             CnnBlock(128, 256),
#
#             # pool3
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#
#             # conv4
#             CnnBlock(256, 512),
#             CnnBlock(512, 512)  # shortcut to up-conv1
#
#         )
#
#         self.block2 = nn.Sequential(
#
#             # pool4
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#
#             # conv5
#             CnnBlock(512, 512),
#             CnnBlock(512, 512)  # shortcut to up-conv2
#
#         )
#
#         self.block3 = nn.Sequential(
#
#             # pool5
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#
#             # conv6
#             CnnBlock(512, 1024),
#             CnnBlock(1024, 1024)  # shortcut to up-conv3
#         )
#
#         self.block4 = UpConvBlock(1024, 1024, (224, 332))
#
#         self.block5 = UpConvBlock(512, 512, (224, 332))
#
#         self.block6 = UpConvBlock(512, 512, (224, 332))
#
#         self.block7 = nn.Sequential(
#             CnnBlock(2048, 1024),
#             nn.Conv2d(1024, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 classes
#             nn.Softmax2d()
#         )
#
#     def forward(self, x, i):
#         block1out = self.block1(x)
#         block2out = self.block2(block1out)
#         block3out = self.block3(block2out)
#
#         # upconvolution
#         block4out = self.block4(block3out)
#         block5out = self.block5(block2out)
#         block6out = self.block6(block1out)
#
#         # concatenate results
#         concatenated = torch.cat((block4out, block5out, block6out), dim=1)  # channels are the second dimension
#
#         block7out = self.block7(concatenated)
#
#         if i == 0:
#             logging.info(f'block1out.shape:{block1out.shape}')
#             logging.info(f'block2out.shape:{block2out.shape}')
#             logging.info(f'block3out.shape:{block3out.shape}')
#             logging.info(f'block4out.shape:{block4out.shape}')
#             logging.info(f'block5out.shape:{block5out.shape}')
#             logging.info(f'block6out.shape:{block6out.shape}')
#             logging.info(f'concatenated.shape:{concatenated.shape}')
#             logging.info(f'block7out.shape:{block7out.shape}')
#
#         return block7out
#

def _build_block(layers_list, layers_dict):
    block = nn.Sequential()

    for layer in layers_list:
        if 'cnn' in layer:

            in_channels = layers_dict[layer][layer + '.in_channels']
            out_channels = layers_dict[layer][layer + '.out_channels']
            kernel_size = layers_dict[layer][layer + '.kernel_size']
            stride = layers_dict[layer][layer + '.stride']
            padding = layers_dict[layer][layer + '.padding']
            block.add_module(layer, CnnBlock(in_channels, out_channels, kernel_size, stride, padding))

        elif 'maxpool' in layer:
            kernel_size = layers_dict[layer][layer + '.kernel_size']
            stride = layers_dict[layer][layer + '.stride']
            padding = layers_dict[layer][layer + '.padding']
            block.add_module(layer, nn.MaxPool2d(kernel_size, stride, padding))

        elif 'linear' in layer:
            in_features = layers_dict[layer][layer + '.in_features']
            out_features = layers_dict[layer][layer + '.out_features']
            batchnorm = layers_dict[layer][layer + '.batchnorm']
            activation = layers_dict[layer][layer + '.activation']
            block.add_module(layer, LinearBlock(in_features, out_features, batchnorm, activation))

        elif 'upconv' in layer:
            in_channels = layers_dict[layer][layer + '.in_channels']
            out_channels = layers_dict[layer][layer + '.out_channels']
            width = layers_dict[layer][layer + '.width']
            height = layers_dict[layer][layer + '.height']
            mode = layers_dict[layer][layer + '.mode']
            block.add_module(layer, UpConvBlock(in_channels, out_channels, (height, width), mode))

        elif 'flatten' in layer:
            block.add_module(layer, nn.Flatten())

        # elif 'lazylinear' in layer:
        #     out_features = layers_dict[layer][layer + '.out_features']
        #     block.add_module(layer, nn.LazyLinear(out_features))  # requires torch version 1.8+
        #     block.add_module('lazybn', nn.BatchNorm1d(out_features))
        #     block.add_module('lazyrelu', nn.ReLU(inplace=True))

        elif 'softmax' in layer:
            block.add_module(layer, nn.Softmax())

    return block


# contains 7 blocks
# concatenate upsampling layers
# block contents are customizable
class SegmentationNetwork2(nn.Module):
    def __init__(self, layers_lists, layers_dict):
        super().__init__()

        # unpack list of layers into 7 blocks
        # each block ends when list entry is BLOCKEND
        complete = 0
        block_list = []
        all_lists = []
        while complete < 7:
            layer = layers_lists.pop(0)  # remove first element
            if layer != 'BLOCKEND':
                block_list.append(layer)
            else:
                # block is complete
                all_lists.append(block_list)

                # start next block
                complete += 1
                block_list = []

        block1_list = all_lists[0]
        block2_list = all_lists[1]
        block3_list = all_lists[2]
        block4_list = all_lists[3]
        block5_list = all_lists[4]
        block6_list = all_lists[5]
        block7_list = all_lists[6]
        logging.info(f'block1_list:{block1_list}')

        # build block 1
        self.block1 = _build_block(block1_list, layers_dict)

        # build block 2
        self.block2 = _build_block(block2_list, layers_dict)

        # build block 3
        self.block3 = _build_block(block3_list, layers_dict)

        # build block 4 - upsampling from block 3
        self.block4 = _build_block(block4_list, layers_dict)

        # build block 5 - upsampling from block 2
        self.block5 = _build_block(block5_list, layers_dict)

        # build block 6 - upsampling from block 1
        self.block6 = _build_block(block6_list, layers_dict)

        # build block 7 - after concatenation of blocks 4,5,6
        self.block7 = _build_block(block7_list, layers_dict)

    def forward(self, x, i):
        block1out = self.block1(x)
        block2out = self.block2(block1out)
        block3out = self.block3(block2out)

        # upconvolution
        block4out = self.block4(block3out)
        block5out = self.block5(block2out)
        block6out = self.block6(block1out)

        # concatenate channels of results
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
    def __init__(self, in_features=4):
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
