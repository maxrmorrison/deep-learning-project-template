import argparse
import shutil
from pathlib import Path

import NAME


###############################################################################
# Entry point
###############################################################################


def main(config, dataset, gpus=None):
    # Create output directory
    directory = NAME.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    NAME.train.run(
        dataset,
        directory,
        directory,
        directory,
        gpus)

    # Evaluate
    NAME.evaluate.datasets([dataset], directory, gpus)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        default=NAME.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The dataset to train on')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


main(**vars(parse_args()))
