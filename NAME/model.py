import pytorch_lightning as pl
import torch
import torch.nn as nn


###############################################################################
# Model
###############################################################################


class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # TODO - define model

    def forward(self):
        """Perform model inference"""
        # TODO
        raise NotImplementedError

    ###########################################################################
    # PyTorch Lightning - step model hooks
    ###########################################################################

    def training_step(self, batch, index):
        """Performs one step of training"""
        # TODO
        raise NotImplementedError

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        # TODO
        raise NotImplementedError

    def test_step(self, batch, index):
        """Performs one step of testing"""
        # OPTIONAL - only implement if you have meaningful objective metrics
        raise NotImplementedError

    ###########################################################################
    # PyTorch Lightning - optimizer
    ###########################################################################

    def configure_optimizer(self):
        """Configure optimizer for training"""
        return torch.optim.Adam(self.parameters())
