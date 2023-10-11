import contextlib
import functools
import os

import torch
import torchutil

import NAME


###############################################################################
# Train
###############################################################################


@torchutil.notify.on_return('train')
def train(datasets, directory=NAME.RUNS_DIR / NAME.CONFIG):
    """Train a model"""
    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(NAME.RANDOM_SEED)
    train_loader = NAME.data.loader(datasets, 'train')
    valid_loader = NAME.data.loader(datasets, 'valid')

    #################
    # Create models #
    #################

    model = NAME.Model()

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = torchutil.checkpoint.latest_path(directory)

    if path is not None:

        # Load model
        model, optimizer, state = torchutil.checkpoint.load(
            path,
            model,
            optimizer)
        step = state['step']

    else:

        # Train from scratch
        step = 0

    ####################
    # Device placement #
    ####################

    import accelerate
    accelerator = accelerate.Accelerator(mixed_precision='fp16')
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        valid_loader)

    #########
    # Train #
    #########

    # Setup progress bar
    progress = NAME.iterator(
        range(step, NAME.STEPS),
        f'Training {NAME.CONFIG}',
        step,
        NAME.STEPS)

    while step < NAME.STEPS:

        for batch in train_loader:

            # TODO - Unpack batch
            () = batch

            # Forward pass
            () = model(
                # TODO - args
            )

            # Compute loss
            losses = loss(
                # TODO - args
            )

            ##################
            # Optimize model #
            ##################

            # Zero gradients
            optimizer.zero_grad()

            # Backward pass
            losses.backward()

            # Update weights
            optimizer.step()

            ############
            # Evaluate #
            ############

            if step % NAME.LOG_INTERVAL == 0:
                evaluate_fn = functools.partial(
                    evaluate,
                    directory,
                    step,
                    model.
                    accelerator)
                evaluate_fn('train', train_loader)
                evaluate_fn('valid', valid_loader)

            ###################
            # Save checkpoint #
            ###################

            if step and step % NAME.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'{step:08d}.pt',
                    model,
                    optimizer,
                    accelerator=accelerator,
                    step=step)

            if step >= NAME.STEPS:
                break

            # Update progress bar
            progress.update()

            # Update training step count
            step += 1

    # Close progress bar
    progress.close()

    # Save final model
    torchutil.checkpoint.save(
        directory / f'{step:08d}.pt',
        model,
        optimizer,
        accelerator=accelerator,
        step=step)


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, accelerator, condition, loader):
    """Perform model evaluation"""
    # Setup evaluation metrics
    metrics = NAME.evaluate.Metrics()

    # Prepare model for inference
    with NAME.inference_context(model) as model:

        for i, batch in enumerate(loader):

            # TODO - unpack batch
            () = batch

            # Forward pass
            () = model(
                # TODO - args
            )

            # Update metrics
            metrics.update(
                # TODO - args
            )

            # Stop when we exceed some number of batches
            if i + 1 == NAME.LOG_STEPS:
                break

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    torchutil.tensorboard.update(directory, step, scalars=scalars)


###############################################################################
# Loss function
###############################################################################


def loss(logits, target):
    """Compute loss function"""
    # TODO
    pass


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(
    rank,
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus[rank])


@contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)

    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()
