"""
All things related to datasets that require customized classes for Training and Validation data.
"""
__author__ = 'ryanquinnnelson'

import logging

import numpy as np
import torchvision.transforms as transforms


def _compose_transforms(transforms_list):
    """
    Build a composition of transformations to perform on image data.
    Args:
        transforms_list (List): list of strings representing individual transformations,
        in the order they should be performed
    Returns: transforms.Compose object containing all desired transformations
    """
    t_list = []

    for each in transforms_list:
        if each == 'RandomHorizontalFlip':
            t_list.append(transforms.RandomHorizontalFlip(0.1))  # customized because 0.5 is too much
        elif each == 'ToTensor':
            t_list.append(transforms.ToTensor())
        elif each == 'Resize':
            t_list.append(transforms.Resize((775, 522), interpolation='bilinear'))

    composition = transforms.Compose(t_list)

    return composition


class ImageDatasetHandler:
    def __init__(self, data_dir,
                 train_dir, train_target_dir,
                 val_dir, val_target_dir,
                 test_dir, test_target_dir,
                 dataset_class, train_transforms):
        """
        Initialize NumericalDatasetHandler.

        :param data_dir (str): fully qualified path to root directory inside which data subdirectories are placed
        :param train_data (str): fully qualified path to training data
        :param val_data (str): fully qualified path to validation data
        :param train_class (Dataset): torch Dataset class to use for training data
        :param val_class (Dataset): torch Dataset class to use for validation data
        """

        logging.info('Initializing image dataset handler...')

        self.data_dir = data_dir
        self.train_dir = train_dir
        self.train_target_dir = train_target_dir
        self.val_dir = val_dir
        self.val_target_dir = val_target_dir
        self.test_dir = test_dir
        self.test_target_dir = test_target_dir
        self.dataset_class = dataset_class
        self.train_transforms = train_transforms

        # determine whether normalize transform should also be applied to validation and test data
        self.should_normalize_val = True if 'Normalize' in train_transforms else False
        self.should_normalize_test = True if 'Normalize' in train_transforms else False

    def get_train_dataset(self):
        """
        Load training data into memory and initialize the Dataset object.
        :return: Dataset
        """

        # initialize dataset
        dataset = self.dataset_class(self.train_dir, self.train_target_dir, self.train_transforms)
        logging.info(f'Loaded {len(dataset)} training images.')
        return dataset

    def get_val_dataset(self):
        """
        Load validation data into memory and initialize the Dataset object.
        :return: Dataset
        """

        if self.should_normalize_val:
            logging.info('Normalizing validation data to match normalization of training data...')
            t = _compose_transforms(['ToTensor', 'Normalize'])
        else:
            t = _compose_transforms(['ToTensor'])

        # initialize dataset
        dataset = self.dataset_class(self.val_dir, self.val_target_dir, t)
        logging.info(f'Loaded {len(dataset)} validation images.')
        return dataset

    def get_test_dataset(self):
        """
        Load validation data into memory and initialize the Dataset object.
        :return: Dataset
        """

        if self.should_normalize_test:
            logging.info('Normalizing validation data to match normalization of training data...')
            t = _compose_transforms(['ToTensor', 'Normalize'])
        else:
            t = _compose_transforms(['ToTensor'])

        # initialize dataset
        dataset = self.dataset_class(self.test_dir, self.test_target_dir, t)
        logging.info(f'Loaded {len(dataset)} test images.')
        return dataset
