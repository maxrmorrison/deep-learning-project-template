import json

import torch
import torchutil

import NAME


###############################################################################
# Evaluate
###############################################################################


@torchutil.notify.on_return('evaluate')
def datasets(
    datasets=NAME.EVALUATION_DATASETS,
    checkpoint=NAME.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Per-file metrics
    file_metrics = NAME.evaluate.Metrics()

    # Per-dataset metrics
    dataset_metrics = NAME.evaluate.Metrics()

    # Aggregate metrics over all datasets
    aggregate_metrics = NAME.evaluate.Metrics()

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = NAME.iterator(
            NAME.data.loader(dataset, 'test'),
            f'Evaluating {NAME.CONFIG} on {dataset}')

        # Iterate over test set
        for batch in iterator:

            # Reset file metrics
            file_metrics.reset()

            # TODO - unpack
            () = batch

            # TODO - copy to device

            # TODO - inference

            # Update metrics
            args = (
                # TODO - args
            )
            file_metrics.update(*args)
            dataset_metrics.update(*args)
            aggregate_metrics.update(*args)

            # Save results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = NAME.EVAL_DIR / NAME.CONFIG
    directory.mkdir(exist_ok=True, parents=True)
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
