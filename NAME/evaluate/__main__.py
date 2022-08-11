"""__main__.py - entry point for NAME.evaluate"""


import argparse
from pathlib import Path

import NAME


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=NAME.DEFAULT_CHECKPOINT,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_args()


if __name__ == '__main__':
    NAME.evaluate.datasets(**vars(parse_args()))
