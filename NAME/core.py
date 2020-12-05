from pathlib import Path


__all__ = ['ASSETS_DIR', 'CACHE_DIR', 'DATA_DIR']


###############################################################################
# Constants
###############################################################################


# Static directories
ASSETS_DIR = Path(__file__).parent / 'assets'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path(__file__).parent.parent / 'data'
