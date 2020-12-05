import argparse


###############################################################################
# Infer
###############################################################################


def from_file(input_file, checkpoint_file):
    """Run inference on one example on disk"""
    # TODO - load and call one()
    raise NotImplementedError


def from_file_to_file(input_file, output_file, checkpoint_file):
    """Run inference on one example on disk and save to disk"""
    # Load and run inference
    result = from_file(input_file)

    # TODO - save to disk
    raise NotImplementedError


def one(input, checkpoint_file):
    """Run inference on one example"""
    # TODO - collate, place on device, and infer
    raise NotImplementedError


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path, help='The input file')
    parser.add_argument('output_file', type=Path, help='File to save results')
    parser.add_argument('checkpoint_file', type=Path, help='Model weight file')
    return parser.parse_args()


if __name__ == '__main__':
    from_file_to_file(**vars(parse_args()))
