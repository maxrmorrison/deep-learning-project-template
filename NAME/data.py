"""data.py - data loading"""


import abc
import itertools
import json
import os

import pytorch_lightning as pl
import torch

import NAME


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The name of the data partition
    """

    def __init__(self, name, partition):
        # Get list of stems
        self.stems = partitions(name)[partition]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # TODO - Load from stem
        raise NotImplementedError

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)


###############################################################################
# Data module
###############################################################################


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module

    Arguments
        name - string
            The name of the dataset
        batch_size - int
            The size of a batch
        num_workers - int or None
            Number data loading jobs to launch. If None, uses num cpu cores.
    """

    def __init__(self, name, batch_size=64, num_workers=None):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """Retrieve the PyTorch DataLoader for training"""
        # TODO - second argument must be the name of your train partition
        return loader(self.name, 'train', self.batch_size, self.num_workers)

    def val_dataloader(self):
        """Retrieve the PyTorch DataLoader for validation"""
        # TODO - second argument must be the name of your valid partition
        return loader(self.name, 'valid', self.batch_size, self.num_workers)

    def test_dataloader(self):
        """Retrieve the PyTorch DataLoader for testing"""
        # TODO - second argument must be the name of your test partition
        return loader(self.name, 'test', self.batch_size, self.num_workers)


###############################################################################
# Data loader
###############################################################################


def loader(dataset, partition, batch_size=64, num_workers=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=Dataset(dataset, partition),
        batch_size=batch_size,
        shuffle='train' in partition,
        num_workers=os.cpu_count() if num_workers is None else num_workers,
        pin_memory=True,
        collate_fn=collate_fn)


###############################################################################
# Collate function
###############################################################################


def collate_fn(batch):
    """Turns __getitem__ output into a batch ready for inference

    Arguments
        batch - list
            The outputs of __getitem__ for each item in batch

    Returns
        collated - tuple
            The input features and ground truth targets ready for inference
    """
    # TODO - Perform any necessary padding or slicing to ensure that input
    #        features and output targets can be concatenated. Then,
    #        concatenate them and return them as torch tensors. See
    #        https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    #        for more information on the collate function (note that
    #        automatic batching is enabled).
    raise NotImplementedError


###############################################################################
# Base dataset metadata
###############################################################################


class Metadata(abc.ABC):
    """Abstract base class for dataset metadata"""

    ###########################################################################
    # File access
    ###########################################################################

    @classmethod
    def files(cls, directory, partition=None):
        """Retrieve filenames in a dataset

        Arguments
            directory - Path
                The root directory of the dataset
            partition - string or None
                Returns files within a partition. If None, returns all files.

        Returns
            files - list(Path)
                The files in the dataset
        """
        # Get stems
        stems = cls.stems(partition)

        # Convert to files
        return [cls.stem_to_file(directory, stem) for stem in stems]

    @classmethod
    def partition_file(cls):
        """Retrieve the name of the partition file

        Returns
            file - Path
                The partition file
        """
        return NAME.ASSETS_DIR / 'partition' / f'{cls.name}.json'

    @classmethod
    def partitions(cls):
        """Get split of stems into partitions

        Returns
            partitions - dict(string, list(string))
                The dictionary of partition names and corresponding stems
        """
        with open(cls.partition_file()) as file:
            return json.load(file)

    @classmethod
    def stems(cls, partition=None):
        """Retrieve the stems in a dataset

        Arguments
            partition - string or None
                Returns stems within a partition. If None, returns all stems.

        Returns
            stems - list(string)
                The stems in the dataset
        """
        # Get partitions
        partitions = cls.partitions()

        # Return all stems
        if partition is None:
            return itertools.chain(*partitions.values())

        # Return stems of a given partition
        return partitions[partition]

    ###########################################################################
    # Conversions
    ###########################################################################

    @staticmethod
    @abc.abstractmethod
    def file_to_stem(file):
        """Convert file to stem"""
        pass

    @staticmethod
    @abc.abstractmethod
    def stem_to_file(directory, stem):
        """Convert stem to file"""
        pass


###############################################################################
# Derived dataset metadata
###############################################################################


# TODO - create a Metadata class for each dataset
class DATASETMetadata(Metadata):

    name = 'DATASET'

    @staticmethod
    def file_to_stem(file):
        """Convert file to stem

        Arguments
            file - Path
                The file to convert

        Returns
            stem - string
                The corresponding stem
        """
        # TODO - define the conversion from a filename to a stem
        raise NotImplementedError

    @staticmethod
    def stem_to_file(directory, stem):
        """Convert stem to file

        Arguments
            directory - Path
                The root directory of the dataset
            stem - string
                The stem to convert

        Returns
            file - Path
                The corresponding file
        """
        # TODO - define the conversion from a stem to a filename
        raise NotImplementedError


###############################################################################
# Functional metadata interface - file access
###############################################################################


def files(name, directory, partition=None):
    """Retrieve filenames in a dataset

    Arguments
        name - string
            The name of the dataset
        directory - Path
            The root directory of the dataset
        partition - string or None
            Returns files within a partition. If None, returns all files.

    Returns
        files - list(Path)
            The files in the dataset
    """
    return metadata(name).files(directory, partition)


def partition_file(name):
    """Retrieve the name of the partition file

    Arguments
        name - string
            The name of the dataset

    Returns
        file - Path
            The partition file
    """
    return metadata(name).partition_file()


def partitions(name):
    """Get split of stems into partitions

    Arguments
        name - string
            The name of the dataset

    Returns
        partitions - dict(string, list(string))
            The dictionary of partition names and corresponding stems
    """
    return metadata(name).partitions()


def stems(name, partition=None):
    """Retrieve the stems in a dataset

    Arguments
        name - string
            The name of the dataset
        partition - string or None
            Returns stems within a partition. If None, returns all stems.

    Returns
        stems - list(string)
            The stems in the dataset
    """
    return metadata(name).stems(partition)


###############################################################################
# Functional metadata interface - conversions
###############################################################################


def file_to_stem(name, file):
    """Convert file to stem

    Arguments
        name - string
            The name of the dataset
        file - Path
            The file to convert

    Returns
        stem - string
            The corresponding stem
    """
    return metadata(name).file_to_stem(file)


def stem_to_file(name, directory, stem):
    """Convert stem to file

    Arguments
        name - string
            The name of the dataset
        directory - Path
            The root directory of the dataset
        stem - string
            The stem to convert

    Returns
        file - Path
            The corresponding file
    """
    return metadata(name).stem_to_file(directory, stem)


###############################################################################
# Utilities
###############################################################################


def metadata(name):
    """Get the metadata for the dataset

    Arguments
        name - string
            The name of the dataset

    Returns
        metadata - Metadata
            The dataset metadata
    """
    # TODO - replace with your datasets
    if name == 'DATASET':
        return DATASETMetadata
    raise ValueError(f'Dataset {name} is not defined')
