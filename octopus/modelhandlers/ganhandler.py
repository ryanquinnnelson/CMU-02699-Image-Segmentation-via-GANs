"""
Handler for VAE models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models import gan


class GanHandler:

    def __init__(self, model_type, layers_list, layers_dict):
        logging.info('Initializing GAN handler...')

        self.model_type = model_type
        self.layers_list = layers_list
        self.layers_dict = layers_dict
        logging.info(f'layers_lists:{self.layers_list}')

    def get_model(self):
        generator, discriminator = None, None

        if self.model_type == 'FlexGAN':
            generator = gan.SegmentationNetwork2(self.layers_list, self.layers_dict)
            discriminator = gan.EvaluationNetwork(4)
        logging.info(f'Model1 initialized:\n{generator}')
        logging.info(f'Model2 initialized:\n{discriminator}')
        return generator, discriminator
