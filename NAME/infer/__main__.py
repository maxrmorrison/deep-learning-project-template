"""__main__.py - entry point for NAME.infer"""


import argparse
from pathlib import Path

import NAME


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
    NAME.infer.from_file_to_file(**vars(parse_args()))
