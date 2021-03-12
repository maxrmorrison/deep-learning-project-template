"""infer.py - model inference"""


###############################################################################
# Infer
###############################################################################


def from_file(input_file, checkpoint_file):
    """Run inference on one example on disk

    Arguments
        input_file - string
            The file containing input data
        checkpoint_file - string
            The model checkpoint file

    Returns
        TODO - define the return value for your project
    """
    # TODO - load input and run inference()
    raise NotImplementedError


def from_file_to_file(input_file, output_file, checkpoint_file):
    """Run inference on one example on disk and save to disk

    Arguments
        input_file - string
            The file containing input data
        output_file - string
            The file to write inference results
        checkpoint_file - string
            The model checkpoint file
    """
    # Load and run inference
    result = from_file(input_file, checkpoint_file)

    # TODO - save to disk
    raise NotImplementedError
