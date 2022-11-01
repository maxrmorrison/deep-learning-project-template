import contextlib
import os

import torch
import tqdm


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def chdir(directory):
    """Context manager for changing the current working directory"""
    curr_dir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curr_dir)


@contextlib.contextmanager
def inference_context(model, device_type):
    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Automatic mixed precision
        with torch.autocast(device_type):

            yield model

    # Prepare model for training
    model.train()


def iterator(iterable, message, total=None):
    """Create a tqdm iterator"""
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        total=len(iterable) if total is None else total)
