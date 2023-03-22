from pathlib import Path

import pytest

import NAME


TEST_ASSETS_DIR = Path(__file__).parent / 'assets'


###############################################################################
# Pytest fixtures
###############################################################################


@pytest.fixture(scope='session')
def dataset():
    """Preload the dataset"""
    return NAME.data.Dataset('DATASET', 'valid')


@pytest.fixture(scope='session')
def model():
    """Preload the model"""
    return NAME.Model()
