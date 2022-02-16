"""
All things related to model training phases.
"""
__author__ = 'ryanquinnnelson'

import logging
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')


class Training:
    """
    Defines object to manage Training phase of training.
    """

    def __init__(self, train_loader, criterion, devicehandler):
        """
        Initialize Training object.

        Args:
            train_loader (DataLoader): DataLoader for training data
            criterion (class): loss function
            devicehandler (DeviceHandler):manages device on which training is being run
        """
        logging.info('Loading training phase...')
        self.train_loader = train_loader
        self.criterion = criterion
        self.devicehandler = devicehandler

    def train_model(self, epoch, num_epochs, model, optimizer):
        """
        Executes one epoch of training.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained
            optimizer (nn.optim): optimizer for this model

        Returns: float representing average training loss

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')

        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
            logging.info(f'training batch:{i}')
            # prep
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

            # compute forward pass
            out = model.forward(inputs, i)

            if i == 0:
                logging.info(f'inputs.shape:{inputs.shape}')
                logging.info(f'targets.shape:{targets.shape}')
                logging.info(f'out.shape:{out.shape}')

            # calculate loss
            loss = self.criterion(out, targets)
            train_loss += loss.item()

            # delete mini-batch data from device
            del inputs
            del targets

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer.step()

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss


def _calculate_num_hits(i, inputs, out):

    batch_size = len(inputs)

    # convert to class labels
    # convert out to class labels
    labels_out = out.argmax(axis=0)

    # convert 2D images into 1D vectors
    out = labels_out.cpu().detach().numpy().reshape((batch_size, -1))
    labels_inputs = inputs.cpu().detach().numpy().reshape((batch_size, -1))

    # compare predictions against actual
    # compare lists of max indices and find the number that match
    n_hits = np.sum(labels_out == labels_inputs)

    if i == 0:
        logging.info(f'reshaped out.shape:{out.shape}')
        logging.info(f'reshaped labels_inputs.shape:{labels_inputs.shape}')
        logging.info(f'labels_out.shape:{labels_out.shape}')
        logging.info(f'n_hits:{n_hits}')

    return n_hits


class Evaluation:
    """
    Defines an object to manage the evaluation phase of training.
    """

    def __init__(self, val_loader, criterion, devicehandler):
        """
        Initialize Evaluation object.

        Args:
            val_loader (DataLoader): DataLoader for validation dataset
            criterion (class): loss function
            devicehandler (DeviceHandler): object to manage interaction of model/data and device
        """
        logging.info('Loading evaluation phase...')
        self.val_loader = val_loader
        self.criterion = criterion
        self.devicehandler = devicehandler

    def evaluate_model(self, epoch, num_epochs, model):
        """
        Perform evaluation phase of training.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained

        Returns: Tuple (float,float) representing (val_loss, val_metric)

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        val_loss = 0
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
                logging.info(f'validation batch:{i}')

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # compute forward pass
                out = model.forward(inputs, i)

                if i == 0:
                    logging.info(f'inputs.shape:{inputs.shape}')
                    logging.info(f'targets.shape:{targets.shape}')
                    logging.info(f'out.shape:{out.shape}')

                # calculate loss
                loss = self.criterion(out, targets)
                val_loss += loss.item()

                # calculate accuracy
                num_hits += _calculate_num_hits(i, out, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = num_hits / len(self.val_loader.dataset)

            return val_loss, val_acc


class Testing:
    """
    Defines an object to manage the testing phase of training.
    """

    def __init__(self, test_loader, criterion, devicehandler):
        logging.info('Loading testing phase...')
        self.test_loader = test_loader
        self.criterion = criterion
        self.devicehandler = devicehandler

    def test_model(self, epoch, num_epochs, model):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        val_loss = 0
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.test_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # compute forward pass
                out = model.forward(inputs)

                # calculate loss
                loss = self.criterion.calculate_loss(out, targets)
                val_loss += loss.item()

                # calculate accuracy
                num_hits += _calculate_num_hits(i, out, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.test_loader)  # average per mini-batch
            val_acc = num_hits / len(self.test_loader.dataset)

            return val_loss, val_acc
