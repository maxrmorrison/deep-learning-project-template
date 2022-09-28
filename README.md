# Deep learning project template

Throughout this template, `NAME` is used to refer to the name of the project
and `DATASET` is used to refer to the name of a dataset.


## Installation

Clone this repo. Change all instances of `NAME` to your project name.
Then run `cd NAME && pip install -e .`.

## Usage

### Download data

Complete all TODOs in `data/download/`, then run `python -m NAME.download DATASET`.


### Partition data

Complete all TODOs in `partition/`, then run `python -m NAME.partition
DATASET`.


### Preprocess data

Complete all TODOs in `preprocess/`, then run `python -m NAME.preprocess
DATASET`. All preprocessed data is saved in `data/cache/DATASET`.


### Train

Complete all TODOs in `data/` and `model.py`, then run `python -m NAME.train --config <config> --dataset
DATASET --gpus <gpus>`.


### Evaluate

Complete all TODOs in `evaluate/`, then run `python -m NAME.evaluate
--datasets <datasets> --checkpoint <checkpoint> --gpu <gpu>`.


### Monitor

Run `tensorboard --logdir runs/`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.


### Test

Tests are written using `pytest`. Run `pip install pytest` to install pytest.
Complete all TODOs in `test_model.py` and `test_data.py`, then run `pytest`.
Adding project-specific tests for preprocessing, inference, and inference is
encouraged.


## FAQ

### What is the directory `NAME/assets` for?

This directory is for
[_package data_](https://packaging.python.org/guides/distributing-packages-using-setuptools/#package-data).
When you pip install a package, pip will
automatically copy the python files to the installation folder (in
`site_packages`). Pip will _not_ automatically copy files that are not Python
files. So if your code depends on non-Python files to run (e.g., a pretrained
model, normalizing statistics, or data partitions), you have to manually
specify these files in `setup.py`. This is done for you in this repo. In
general, only small files that are essential at runtime should be placed in
this folder.


### What if my evaluation includes subjective experiments?

In this case, replace the `<file>` argument of `NAME.evaluate` with a
directory. Write any objective metrics to a file within this directory, as well
as any generated files that will be subjectively evaluated.


### How do I release my code so that it can be downloaded via pip?

Code release involves making sure that `setup.py` is up-to-date and then
uploading your code to [`pypi`](https://www.pypi.org).
[Here](https://packaging.python.org/tutorials/packaging-projects/) is a good
tutorial for this process.
