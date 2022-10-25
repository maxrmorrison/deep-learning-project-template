"""__main__.py - entry point for NAME.preprocess"""


import argparse

import NAME


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'datasets',
        nargs='+',
        help='The name of the datasets to preprocess')
    return parser.parse_args()


NAME.preprocess.datasets(**vars(parse_args()))
