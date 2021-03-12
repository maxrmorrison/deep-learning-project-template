"""core.py - model evaluation"""


import json

import NAME


###############################################################################
# Evaluate
###############################################################################


def dataset(name, partition, model):
    """Evaluate a dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The partition to evaluate

    Returns
        results - dict
            The dictionary of results. The key is the name of a metric and
            the value is the value received for that metric. Must be JSON
            serializable.
    """
    # Get stems for this partition
    stems = NAME.data.partitions(name)[partition]

    # Resolve stems to filenames
    files = [NAME.data.stem_to_file(name, stem) for stem in stems]

    # Partition files
    return from_files(model, files)


def dataset_to_file(name, partition, model, file):
    """Evaluate dataset and write results to json file

    Arguments
        name - string
            The name of the dataset
        partition - string
            The partition to evaluate
        model - NAME.Model
            The model to evaluate
        file - Path
            The json file to save results to
    """
    with open(file, 'w') as file:
        json.dump(dataset(name, partition, model), file)


def from_files(model, files):
    """Evaluate files

    Arguments
        model - NAME.Model
            The model to evaluate
        files - list(string)
            The files to evaluate

    Returns
        results - dict
            The dictionary of results. The key is the name of a metric and
            the value is the value received for that metric. Must be JSON
            serializable.
    """
    # TODO - evaluate
    raise NotImplementedError
