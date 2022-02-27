import logging

import torch.nn as nn
import torch


class ModelHandler:
    def __init__(self):
        pass

    # TODO: revise models to modular, accept argument for which model type to use
    def get_models(self, wandb_config):
        sn = None
        en = None

        if wandb_config.sn_model_type == 'SNLite':
            sn = SNLite()
        elif wandb_config.sn_model_type == 'ConcatenationFCN':
            num_fcn_blocks = wandb_config.num_fcn_blocks
            block_depth = wandb_config.block_depth
            input_channels = wandb_config.input_channels
            output_channels = wandb_config.output_channels
            first_layer_out_channels = wandb_config.first_layer_out_channels
            block_pattern = wandb_config.block_pattern
            upsampling_pattern = wandb_config.upsampling_pattern
            original_height = wandb_config.original_height
            original_width = wandb_config.original_width

            sn = ConcatenationFCN(num_fcn_blocks, block_depth, input_channels, output_channels,
                                  first_layer_out_channels, block_pattern, upsampling_pattern, original_height,
                                  original_width)

        elif wandb_config.sn_model_type == 'ZhengSN':

            input_channels = wandb_config.input_channels
            sn = ZhengSN(input_channels)

        if wandb_config.en_model_type == 'ENLite':
            en = ENLite()
        elif wandb_config.sn_model_type == 'ZhengEN':
            en = ZhengEN()

        logging.info(f'Generator model initialized:\n{sn}')
        logging.info(f'Discriminator model initialized:\n{en}')

        return [sn, en], ['sn_model', 'en_model']


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

    def __init__(self, block_number, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnn_block = nn.Sequential()
        self.cnn_block.add_module('cnn' + str(block_number) + '_0',
                                  nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, bias=False))
        self.cnn_block.add_module('bn' + str(block_number) + '_0', nn.BatchNorm2d(out_channels))
        self.cnn_block.add_module('relu' + str(block_number) + '_0', nn.ReLU(inplace=True))

        # initialize weights
        self.cnn_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'cnn_block_input:{x.shape}')
        x = self.cnn_block(x)
        # logging.info(f'cnn_block:{x.shape}')
        return x


class UpConvBlock(nn.Module):
    # bi-linear interpolation, or learned up-sampling filters
    # nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear')
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

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


# Simplified SN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation..."
# layers are reduced to run fast for testing purposes
class SNLite(nn.Module):
    def __init__(self):
        super().__init__()

        self.block7 = nn.Sequential(
            CnnBlock(0, 3, 3),
            nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):
        block7out = self.block7(x)

        if i == 0:
            logging.info(f'block7out.shape:{block7out.shape}')

        return block7out


# Simplified EN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation..."
# layers are reduced to run fast for testing purposes
class ENLite(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(

            # conv1
            CnnBlock(0, 4, 4)

        )

        self.block2 = nn.Sequential(

            nn.Flatten(),  # need to convert 2d to 1d

        )

        self.block3 = nn.Sequential(
            LinearBlock(297472, 256),  # 4*224*332
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


# SN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation Utilizing Unannotated Images"
class ZhengSN(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block1 = nn.Sequential(
            # conv1
            CnnBlock(1, in_features, 64),
            CnnBlock(2, 64, 64),

            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv2
            CnnBlock(3, 64, 128),

            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv3
            CnnBlock(4, 128, 128),
            CnnBlock(5, 128, 256),

            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv4
            CnnBlock(6, 256, 512),
            CnnBlock(7, 512, 512)  # shortcut to up-conv1

        )

        self.block2 = nn.Sequential(

            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv5
            CnnBlock(8, 512, 512),
            CnnBlock(9, 512, 512)  # shortcut to up-conv2

        )

        self.block3 = nn.Sequential(

            # pool5
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv6
            CnnBlock(10, 512, 1024),
            CnnBlock(11, 1024, 1024)  # shortcut to up-conv3
        )

        self.block4 = UpConvBlock(1024, 1024, (224, 332))

        self.block5 = UpConvBlock(512, 512, (224, 332))

        self.block6 = UpConvBlock(512, 512, (224, 332))

        self.block7 = nn.Sequential(
            CnnBlock(12, 2048, 1024),
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


# EN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation Utilizing Unannotated Images"
class ZhengEN(nn.Module):
    def __init__(self, en_input_features=4):
        super().__init__()

        self.block1 = nn.Sequential(

            # conv1
            CnnBlock(1, en_input_features, 64),
            CnnBlock(2, 64, 64),

            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv2
            CnnBlock(3, 64, 128),

            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv3
            CnnBlock(4, 128, 256),
            CnnBlock(5, 256, 256),

            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv4
            CnnBlock(6, 256, 512),
            CnnBlock(7, 512, 512),

            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv5
            CnnBlock(8, 512, 512),
            CnnBlock(9, 512, 512)

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


def _generate_channels_lists(in_channels, block_pattern, block_depth):
    in_channels_list = []
    out_channels_list = []

    # calculate channels sizes for block based on block pattern
    current_in_channels = in_channels
    current_out_channels = None
    if block_pattern == 'single_run':

        # out_channel is 2x the in_channel of that layer
        # for block_depth = 3
        # in_channels_list  [ 64, 128, 256]
        # out_channels_list [128, 256, 512]
        for each in range(block_depth):
            # in channels
            in_channels_list.append(current_in_channels)  # match output channels of previous layer
            current_out_channels = current_in_channels * 2  # output is 2x the input

            # out channels
            out_channels_list.append(current_out_channels)
            current_in_channels = current_out_channels

    elif block_pattern == 'double_run':

        # odd layers have in_channels and out_channels that are the same value
        # even layers have out_channel = 2 * in_channel
        # for block_depth = 4
        # in_channels_list  [64,  64, 128, 128]
        # out_channels_list [64, 128, 128, 256]
        current_in_channels = in_channels
        current_out_channels = current_in_channels
        is_symmetrical_layer = True
        for each in range(block_depth):

            # in channels
            in_channels_list.append(current_in_channels)  # match output channels of previous layer

            if is_symmetrical_layer:
                current_out_channels = current_in_channels
            else:
                current_out_channels = current_in_channels * 2

            # toggle opposite rule for next layer
            is_symmetrical_layer = not is_symmetrical_layer

            # out channels
            out_channels_list.append(current_out_channels)
            current_in_channels = current_out_channels

    return in_channels_list, out_channels_list


class FcnBlock(nn.Module):

    def __init__(self, block_number, in_channels, pool_layer_first, block_depth, block_pattern,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        #         print('init')

        # calculate sizes for input and output channels for all cnn layers in this block
        in_channels_list, out_channels_list = _generate_channels_lists(in_channels, block_pattern, block_depth)
        #         print(in_channels_list, out_channels_list)

        self.in_channels = in_channels
        self.block_number = block_number
        self.out_channels = out_channels_list[-1]  # last out channel of block

        # build block
        self.fcn_block = nn.Sequential()

        if pool_layer_first:
            self.fcn_block.add_module('pool' + str(block_number), nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # build block one group at a time
        for i, curr_in_channels in enumerate(in_channels_list):
            curr_out_channels = out_channels_list[i]
            self.fcn_block.add_module('cnn' + str(block_number) + '_' + str(i),
                                      nn.Conv2d(curr_in_channels, curr_out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, bias=False))

            self.fcn_block.add_module('bn' + str(block_number) + '_' + str(i), nn.BatchNorm2d(curr_out_channels))
            self.fcn_block.add_module('relu' + str(block_number) + '_' + str(i), nn.ReLU(inplace=True))

        # initialize weights
        self.fcn_block.apply(_init_weights)

    #         print('applied')

    def forward(self, x):
        # logging.info(f'fcn_block_inputx:{x.shape}')
        x = self.fcn_block(x)
        # logging.info(f'fcn_block_input:{x.shape}')
        return x


# TODO: allow prob block to have custom out channels for first cnn layer
# TODO: upsampling allow different input and output channel sizes
# blocks start with pool so output of block can be concatenated, first block doesn't have pool
# DCAN block_pattern: if 6 blocks, blocks 4 and 5 are the same channels

# ZhengFCN: concatenates upsampling channels before calculating the 2 class label probabilities
# DCAN_FCN: calculates 2 class label probabilities for each upsampling channel, then sums probabilities
class ConcatenationFCN(nn.Module):
    def __init__(self, num_fcn_blocks=3, block_depth=1, input_channels=3, output_channels=2,
                 first_layer_out_channels=64,
                 block_pattern='single_run', upsampling_pattern='last_three', original_height=224, original_width=332):
        """

        Args:
            n_blocks:  4,5,6 number of cnn blocks in network
            block_depth: 1, 2, 3, 4 number of cnn layers in a block
            input_channels: 3 channels in original image
            output_channels: 2 number of classes to softmax
            start_channels: 64
            block_pattern: single_run, double_run, dcan_run,
            upsampling_pattern: last_three, last_two
            original_height: 224 upsampling to restore image to this size
            original_width: 332 upsampling to restore image to this size
        """
        super().__init__()

        self.block_pattern = block_pattern
        self.upsampling_pattern = upsampling_pattern

        # add input block
        block_number = 0
        self.input_block = CnnBlock(block_number, input_channels, first_layer_out_channels)

        # add FCN blocks
        fcn_blocks = []
        curr_in_channels = first_layer_out_channels
        pool_layer_first = False  # first FCN block doesn't start with pool

        for n in range(1, num_fcn_blocks + 1):
            # create block
            block_number = n
            block = FcnBlock(block_number, curr_in_channels, pool_layer_first, block_depth, block_pattern)
            fcn_blocks.append(block)

            # update settings for next block
            curr_in_channels = block.out_channels
            pool_layer_first = True  # all additional FCN blocks start with a pooling layer

        # subdivide fcn blocks based on connections to upsampling blocks
        self.fcn1 = None
        self.fcn2 = None
        self.fcn3 = None

        if upsampling_pattern in ['last_three']:
            # three fcn blocks
            self.fcn1 = nn.ModuleList(fcn_blocks[:num_fcn_blocks - 2])
            self.fcn2 = nn.ModuleList([fcn_blocks[-2]])
            self.fcn3 = nn.ModuleList([fcn_blocks[-1]])

        # add upsampling blocks
        self.up1 = None
        self.up2 = None
        self.up3 = None

        block_number += 1
        size = (original_height, original_width)
        concatenated_channels = 0

        if upsampling_pattern == 'last_three':
            self.up1 = UpConvBlock(self.fcn1[-1].out_channels, self.fcn1[-1].out_channels, size)
            self.up2 = UpConvBlock(self.fcn2[-1].out_channels, self.fcn2[-1].out_channels, size)
            self.up3 = UpConvBlock(self.fcn3[-1].out_channels, self.fcn3[-1].out_channels, size)

            # calculate concatenated_channels
            concatenated_channels += self.up1.out_channels
            concatenated_channels += self.up2.out_channels
            concatenated_channels += self.up3.out_channels

        # create probability block
        block_number += 1
        self.map_block = nn.Sequential(
            CnnBlock(block_number, concatenated_channels, concatenated_channels),
            nn.Conv2d(concatenated_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):

        x = self.input_block(x)

        # perform forward pass on fcn and up blocks at the same time to avoid needing extra copies of fcn output
        if self.upsampling_pattern in ['last_three']:

            for fcn in self.fcn1:  # ModuleList requires iteration, can't do forward pass directly
                x = fcn(x)
            up1 = self.up1(x)

            for fcn in self.fcn2:  # ModuleList requires iteration, can't do forward pass directly
                x = fcn(x)
            up2 = self.up2(x)

            for fcn in self.fcn3:  # ModuleList requires iteration, can't do forward pass directly
                x = fcn(x)
            up3 = self.up3(x)

            up_tuple = (up1, up2, up3)

        else:
            raise NotImplementedError

        # concatenate upsampling output
        x = torch.cat(up_tuple, dim=1)  # channels are the second dimension

        x = self.map_block(x)

        return x
