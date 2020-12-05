import argparse
from pathlib import Path

import NAME


###############################################################################
# Preprocess
###############################################################################


def from_dataset_to_files(dataset):
    """Preprocess dataset in data directory and save in cache

    Arguments
        dataset - string
            The name of the dataset to preprocess
    """
    # Get input files to be preprocessed
    # TODO - replace with your datasets
    if dataset == 'DATASET':
        inputs = DATASET_inputs()
    else:
        raise ValueError(f'Dataset {dataset} is not implemented')

    # Preprocess
    from_files_to_files(dataset, inputs)


def from_files_to_files(dataset, files):
    """Preprocess input files in data directory and save in cache

    Arguments
        dataset - string
            The name of the dataset to preprocess
        files - list(string) or list(tuple)
            The dataset-specific inputs for preprocessing
    """
    input_directory = NAME.DATA_DIR / dataset
    output_directory = NAME.CACHE_DIR / dataset

    # TODO - Perform preprocessing. Note that you will need to preprocess
    #        a single example in infer.py. It is recommended to design your
    #        code accordingly (e.g., with a from_file() function that loads,
    #        preprocesses, and returns preprocessed features).
    raise NotImplementedError


###############################################################################
# Dataset-specific
###############################################################################


def DATASET_inputs():
    """Get a list of preprocessing inputs

    Returns
        inputs - list
            A list of filenames or tuples of filenames and metadata that make
            up the preprocessing inputs for each item in the dataset
    """
    # TODO
    raise NotImplementedError


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help='The name of the dataset to preprocess')
    return parser.parse_args()


if __name__ == '__main__':
    from_dataset_to_files(**vars(parse_args()))
