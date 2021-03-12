"""core.py - data preprocessing"""


import NAME


###############################################################################
# Preprocess
###############################################################################


def dataset(name):
    """Preprocess dataset in data directory and save in cache

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    # Get input files and metadata to be preprocessed
    # TODO - replace with your datasets
    if name == 'DATASET':
        inputs = DATASET_inputs()
    else:
        raise ValueError(f'Dataset {name} is not implemented')

    input_directory = NAME.DATA_DIR / name
    output_directory = NAME.CACHE_DIR / name

    # TODO - Perform preprocessing
    raise NotImplementedError


###############################################################################
# Dataset-specific
###############################################################################


def DATASET_inputs():
    """Get a list of preprocessing inputs

    Returns
        inputs - list(tuple)
            Filenames and any metadata needed to preprocess each item in the
            dataset. The exact type of each element is project-specific.
            For image classification, inputs is a list of filenames of images
            to preprocess. For text-to-speech, inputs is a pair of filenames
            (the text file and speech file).
    """
    # TODO
    raise NotImplementedError
