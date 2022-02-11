"""
Handler for VAE models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models import gan


class GanHandler:

    def __init__(self, model_type, input_size):

        logging.info('Initializing GAN handler...')

        self.model_type = model_type
        self.input_size = input_size

    def get_model(self):

        model1, model2 = None, None

        if self.model_type == 'ZhangGAN':
            model1 = gan.SegmentationNetwork(in_features=3, input_size=self.input_size)
            model2 = gan.EvaluationNetwork(self.input_size)
        logging.info(f'Model1 initialized:\n{model1}')
        logging.info(f'Model2 initialized:\n{model2}')
        return model1, model2
