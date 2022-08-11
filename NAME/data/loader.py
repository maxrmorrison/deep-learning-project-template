import torch

import NAME


def loaders(dataset):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train'), loader(dataset, 'valid')


def loader(dataset, partition):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=NAME.data.Dataset(dataset, partition),
        batch_size=NAME.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=NAME.NUM_WORKERS,
        pin_memory=True,
        collate_fn=NAME.data.collate)
