import argparse
import json
from pathlib import Path

import NAME


###############################################################################
# Partition
###############################################################################


def from_dataset(dataset):
    """Partition from dataset files

    Arguments
        dataset - string
            The name of the dataset

    Returns
        partitions - dict(string, list(string))
            The resulting partitions. The key is the partition name and the
            value is the list of stems belonging to that partition.
    """
    # Get a list of filenames without extension to be partitioned
    # TODO - replace with your datasets
    if dataset == 'DATASET':
        stems = DATASET_stems()
    else:
        raise ValueError(f'Dataset {dataset} is not implemented')

    # Partition files
    return from_stems(stems)


def from_dataset_to_file(dataset):
    """Partition and write json file

    Arguments
        dataset - string
            The name of the dataset
    """
    # Create output directory
    output_directory = NAME.ASSETS_DIR / dataset
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write partition file
    with open(output_directory / 'partition.json', 'w') as file:
        json.dump(from_dataset(dataset), file)


def from_stems(stems):
    """Partition stems

    Arguments
        stems - list(string)
            The dataset file stems to partition

    Returns
        partitions - dict(string, list(string))
            The resulting partitions. The key is the partition name and the
            value is the list of stems belonging to that partition.
    """
    # TODO
    raise NotImplementedError


###############################################################################
# Dataset-specific
###############################################################################


def DATASET_stems():
    """Get a list of filenames without extension to be partitioned

    Returns
        stems - list(string)
            The list of file stems to partition
    """
    # TODO
    raise NotImplementedError


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The name of the dataset to partition')
    return parser.parse_args()


if __name__ == '__main__':
    from_dataset_to_file(**vars(parse_args()))
