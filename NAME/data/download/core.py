import urllib
import shutil
import ssl

import NAME


###############################################################################
# Download datasets
###############################################################################


def datasets(datasets=NAME.DATASETS):
    """Download datasets"""
    # TODO - download datasets
    pass


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
            open(file, 'wb') as output:
        shutil.copyfileobj(response, output)
