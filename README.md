# Cotton Counter

## Overview

This project contains reference code for the following paper:

[Weakly-supervised learning to automatically count cotton flowers from aerial imagery](https://www.sciencedirect.com/science/article/pii/S0168169922000515)

## Installing dependencies

This code requires Python 3.8 and [Poetry](https://python-poetry.org/docs/).

To install, run
```
poetry install --no-root
```

## Running Pipelines

This project uses [Kedro](https://kedro.readthedocs.io/en/stable/introduction/introduction.html).
It implements different pipelines for various tasks. A pipeline can be run with:
```
poetry run kedro run --pipeline <pipeline name>
```

The pipelines are:
- `model_training`: Trains the model from scratch using existing data.
- `model_evaluation`: Performs evaluation on a trained model.

### Training

Training requires that you have access to the `TFRecords` files containing the
dataset. These are specified in [`catalog.yml`](conf/base/catalog.yml), under
`tfrecord_train`, `tfrecord_test`, `tfrecord_validate`, and 
`tfrecord_test_alternate`. These are the training, testing, validation, and 
supplemental testing datasets, respectively. (Only the first three are
used for training. The last one is [part C](https://www.sciencedirect.com/science/article/pii/S0168169922000515#t0005)
of the dataset used for extra evaluation.) Modify the `filepath` attribute of
these catalog entries if necessary so that they point to the correct location
on your computer.

Once training is complete, the default location of the trained model is
`output_data/06_models/fully_trained.hd5/`. Trained models will be versioned
in subdirectories.

### Model Evaluation

Model evaluation requires a trained model to already have been saved. It will
generate a variety of reports that are located by default under
`output_data/08_reporting/`. See the [catalog](conf/base/pipelines/model_evaluation/catalog.yml)
for some documentation on these reports.

## Working with Kedro from notebooks

You can use Jupyter notebooks with this project. First, start a local 
notebook server:

```
poetry run kedro jupyter notebook
```

You can also start Jupyter Lab:

```
poetry run kedro jupyter lab
```

And if you want to run an IPython session:

```
poetry run kedro ipython
```

Running Jupyter or IPython this way provides the following variables in
scope: `proj_dir`, `proj_name`, `conf`, `io`, `parameters` and `startup_error`.

### Converting notebook cells to nodes in a Kedro project

Once you are happy with a notebook, you may want to move your code over into the Kedro project structure for the next stage in your development. This is done through a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#cell-tags) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`.
```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. To this end, you can run the following command to convert all notebook files found in the project root directory and under any of its sub-folders.
```
kedro jupyter convert --all
```

### Ignoring notebook output cells in `git`

In order to automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be left intact locally.