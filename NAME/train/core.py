import contextlib
import functools
import os

import torch

import NAME


###############################################################################
# Training interface
###############################################################################


def run(
    datasets,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus=None):
    """Run model training"""
    # Distributed data parallelism
    if gpus and len(gpus) > 1:
        args = (
            datasets,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus)
        torch.multiprocessing.spawn(
            train_ddp,
            args=args,
            nprocs=len(gpus),
            join=True)

    else:

        # Single GPU or CPU training
        train(
            datasets,
            checkpoint_directory,
            output_directory,
            log_directory,
            None if gpus is None else gpus[0])

    # Return path to model checkpoint
    return NAME.checkpoint.latest_path(output_directory)


###############################################################################
# Training
###############################################################################


def train(
    datasets,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpu=None):
    """Train a model"""
    # Get DDP rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = None

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(NAME.RANDOM_SEED)
    train_loader = NAME.data.loader(datasets, 'train', gpu)
    valid_loader = NAME.data.loader(datasets, 'valid', gpu)

    #################
    # Create models #
    #################

    model = NAME.model.Model().to(device)

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank])

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=[.80, .99],
        eps=1e-9)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = NAME.checkpoint.latest_path(
        checkpoint_directory,
        '*.pt')

    if path is not None:

        # Load model
        model, optimizer, step = NAME.checkpoint.load(path, model, optimizer)

    else:

        # Train from scratch
        step = 0

    #####################
    # Create schedulers #
    #####################

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=NAME.LEARNING_RATE_DECAY,
        last_epoch=step // len(train_loader.dataset) if step else -1)
    scheduler = scheduler_fn(optimizer)

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Get total number of steps
    steps = NAME.STEPS

    # Setup progress bar
    if not rank:
        progress = NAME.iterator(
            range(step, steps),
            f'Training {penne.CONFIG}',
            steps)
    while step < steps:

        model.train()
        for batch in train_loader:

            # TODO - Unpack batch
            (
            ) = (item.to(device) for item in batch)

            # Bundle training input
            model_input = (""" TODO - pack network input""")

            with torch.autocast(device.type):

                # Forward pass
                # TODO - unpack network output
                (
                ) = model(*model_input)

                # TODO - compute losses
                losses = 0.

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                ############
                # Evaluate #
                ############

                if step % NAME.LOG_INTERVAL == 0:
                    evaluate_fn = functools.partial(
                        evaluate,
                        log_directory,
                        step,
                        model,
                        gpu)
                    evaluate_fn('train', train_loader)
                    evaluate_fn('valid', valid_loader)

                ###################
                # Save checkpoint #
                ###################

                if step and step % NAME.CHECKPOINT_INTERVAL == 0:
                    NAME.checkpoint.save(
                        model,
                        optimizer,
                        step,
                        output_directory / f'{step:08d}.pt')

            # Update training step count
            if step >= steps:
                break
            step += 1

            # Update progress bar
            if not rank:
                progress.update()

        # Update learning rate every epoch
        scheduler.step()

    # Close progress bar
    if not rank:
        progress.close()

    # Save final model
    NAME.checkpoint.save(
        model,
        optimizer,
        step,
        output_directory / f'{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, gpu, condition, loader):
    """Perform model evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Prepare model for inference
    with NAME.inference_context(model, device.type) as model:

        for i, batch in enumerate(loader):

            # TODO - unpack batch
            () = batch

            # TODO - send to device and forward pass

            # TODO - update metrics

            # Stop when we exceed some number of batches
            if i + 1 == NAME.LOG_STEPS:
                break

    # TODO - write to tensorboard

    # Prepare model for training
    model.train()


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(rank, dataset, directory, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(dataset, directory, gpus)


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
