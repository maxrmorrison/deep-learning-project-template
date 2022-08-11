import json

import promovits


###############################################################################
# Loading utilities
###############################################################################

def partition(dataset):
    """Load partitions for dataset"""
    with open(promovits.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
