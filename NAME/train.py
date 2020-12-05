import argparse
from pathlib import Path

import pytorch_lightning as pl

import NAME


###############################################################################
# Train
###############################################################################


def train():
    """Train a model"""
    parser = argparse.ArgumentParser(add_help=False)

    # Add trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)

    # Add project arguments
    # TODO - If you have one dataset, change the default to that dataset.
    #        Otherwise, delete the default.
    parser.add_argument(
        '--dataset',
        default='DATASET',
        help='The name of the dataset')

    # Parse
    args = parser.parse_args()

    # Setup tensorboard
    logger = pl.loggers.TensorBoardLogger('logs', name=Path().parent.name)

    # Setup data
    datamodule = NAME.DataModule(args.dataset)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    # Train
    trainer.fit(NAME.Model(), datamodule=datamodule)


if __name__ == '__main__':
    train()
