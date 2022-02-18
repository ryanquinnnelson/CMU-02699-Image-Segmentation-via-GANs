"""
All things related to model training phases.
"""
__author__ = 'ryanquinnnelson'

import logging
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')


def _combine_input_and_map(input, map):
    return 0.0


def _d_loss(pred, annotated=True):
    return 0.0


def _g_loss(pred):
    return 0.0


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

    def train_model(self, epoch, num_epochs, g_model, g_optimizer, d_model, d_optimizer, use_gan):
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')

        g_train_loss = 0
        d_train_loss = 0

        # Set model in 'Training mode'
        g_model.train()
        d_model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
            logging.info(f'training batch:{i}')
            # prep
            g_optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, targets = self.devicehandler.move_data_to_device(g_model, inputs, targets)

            # compute forward pass on generator
            out = g_model.forward(inputs, i)

            if i == 0:
                logging.info(f'inputs.shape:{inputs.shape}')
                logging.info(f'targets.shape:{targets.shape}')
                logging.info(f'out.shape:{out.shape}')

            # calculate loss
            g_loss = self.criterion(out, targets)
            g_train_loss += g_loss.item()

            if use_gan:
                # set sigma based on the epoch
                sigma = 0.1
                sigma += (epoch / 30000)

                # compute forward pass on discriminator using fake data
                d_input = _combine_input_and_map(input, out)
                unannotated_pred = d_model(i, d_input)
                d_loss_unannotated = _d_loss(unannotated_pred, annotated=False)

                # compute forward pass on discriminator using real data
                d_input = _combine_input_and_map(inputs, targets)
                annotated_pred = d_model(i, d_input)
                d_loss_annotated = _d_loss(annotated_pred, annotated=True)

                # calculate total discriminator loss for fake and real
                d_loss = sigma * (d_loss_unannotated + d_loss_annotated)
                d_loss.backward()
                d_train_loss += d_loss.item()

                # update discriminator weights
                d_optimizer.step()

                # compute forward pass on updated discriminator using fake data
                d_input = _combine_input_and_map(input, out)
                fake_pred = d_model(i, d_input)

                # g_loss based on discriminator predictions
                # if discriminator predicts some as fake, generator not doing good enough job
                total_g_loss = g_loss(fake_pred)
            else:
                total_g_loss = g_loss

            # compute backward pass of generator
            total_g_loss.backward()

            # update generator weights
            g_optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        g_train_loss /= len(self.train_loader)

        return g_train_loss


def _calculate_num_hits(i, targets, out):
    # convert to class labels
    # convert out to class labels
    labels_out = out.argmax(axis=1)
    if i == 0:
        logging.info(f'labels_out.shape:{labels_out.shape}')

    # compare predictions against actual
    compare = targets == labels_out

    # # convert 2D images into 1D vectors
    # out = labels_out.cpu().detach().numpy().reshape((batch_size, -1))
    # labels_inputs = inputs.cpu().detach().numpy().reshape((batch_size, -1))

    # compare lists of max indices and find the number that match
    n_hits = np.sum(compare.cpu().detach().numpy())

    if i == 0:
        logging.info(f'n_hits:{n_hits}')

    return n_hits


# https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def _calculate_iou_score(i, targets, out):
    targets = targets.cpu().detach().numpy()

    # convert to class labels
    # convert out to class labels
    labels_out = out.argmax(axis=1)
    labels_out = labels_out.cpu().detach().numpy()

    intersection = np.logical_and(targets, labels_out)
    union = np.logical_or(targets, labels_out)

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


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

    def evaluate_model(self, epoch, num_epochs, g_model, d_model):
        """
        Perform evaluation phase of training.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            g_model (nn.Module): model being trained

        Returns: Tuple (float,float) representing (val_loss, val_metric)

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        val_loss = 0
        actual_hits = 0
        score = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            g_model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
                logging.info(f'validation batch:{i}')

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(g_model, inputs, targets)

                # compute forward pass
                out = g_model.forward(inputs, i)

                if i == 0:
                    logging.info(f'inputs.shape:{inputs.shape}')
                    logging.info(f'targets.shape:{targets.shape}')
                    logging.info(f'out.shape:{out.shape}')

                # calculate loss
                loss = self.criterion(out, targets)
                val_loss += loss.item()

                # calculate accuracy
                actual_hits += _calculate_num_hits(i, targets, out)
                score += _calculate_iou_score(i, targets, out)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            possible_hits = (len(self.val_loader.dataset) * 224 * 332)
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = actual_hits / possible_hits
            iou_score = score / len(self.val_loader.dataset)

            return val_loss, val_acc, iou_score


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
