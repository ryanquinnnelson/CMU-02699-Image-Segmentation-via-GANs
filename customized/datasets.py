"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

from torch.utils.data import Dataset
import os
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list.remove('.DS_Store')  # remove mac generated files
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((775, 522), resample=Image.BILINEAR) # standardize image size
        tensor_img = self.transform(img)
        return tensor_img
