"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def _apply_transformations(img, target):
    if random.random() > 0.5:
        # print('vflip')
        img = transforms.functional_pil.vflip(img)
        target = transforms.functional_pil.vflip(target)

    if random.random() > 0.5:
        # print('hflip')
        img = transforms.functional_pil.hflip(img)
        target = transforms.functional_pil.hflip(target)

    return img, target


class ImageDataset(Dataset):

    def __init__(self, img_dir, targets_dir, transform=None):
        self.img_dir = img_dir
        self.targets_dir = targets_dir
        self.transform = transform

        # prepare image list
        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list.remove('.DS_Store')  # remove mac generated files
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])

        # generate target name
        # target image name matches image but also includes suffix
        img_name = self.img_list[idx][:-4]  # strip .bmp
        target_path = os.path.join(self.targets_dir, img_name + '_anno.bmp')

        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        # standardize image size
        img = img.resize((775, 522), resample=Image.BILINEAR)  # standardize image size
        target = target.resize((775, 522), resample=Image.BILINEAR)  # standardize target size

        # apply matching transformations to image and target
        img, target = _apply_transformations(img, target)

        # convert to tensors
        tensor_img = self.transform(img)
        tensor_target = self.transform(target)

        # keep only first channel because all three channels are given the same value
        tensor_target_first_channel = tensor_target[0]

        # convert all nonzero target values to 1
        # nonzero values indicate segment
        # zero values indicate background
        tensor_target_first_channel[tensor_target_first_channel != 0] = 1.0

        # convert target to long datatype to indicate classes
        tensor_target_first_channel = tensor_target_first_channel.to(torch.long)

        return tensor_img, tensor_target_first_channel
