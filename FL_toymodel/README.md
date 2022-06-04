# Federated Learning (FL) toy model
This repo contains a toy model of a training scenario using FL, or more precisely Federated Averaging (FedAvg). 

It is build to run experiments to compare training efficiency of:
- FedAvg training: with a single central server and a set of clients, trained using Stochastic Gradient Descent (SGD).
- Centralized SGD training: the ususal ML workflow, using SGD to train a model.
- Linear regression training: the ususal ML workflow, solving the linear regression problem in one step by solving the normal-equations.

## Data
Data consists of one input and one output, which is generated by an underlying model. This model itself can be chosen to be anything, but is usually set to a polynomial of certain degree. The final step is to add Gaussian white noise to the output.

## Clients
Each client (both FL clients and server are seen as clients):
- Has access to a subset of the data.
- Has a preprocessor that creates a set of polynomial features (of certain degree) of the input data.
- Has a linear model that predicts an output based on the polynomial features coming from the preprocessor.
- Has a SGD training method for iterative training.
- Has a linear regression training method for one-step training.

## Experiments:
- Hyperparameters of experiments can be generated for multiple experiments at once to allow analysis of the effect of certain parameters.
- The hyperparameters that are changed at each sub-experiment are stored inside the `changing_hyperparams.csv` file, which is stored alongisde the experiments.
- Experiments are stored inside subfolders of the `experiment_results` folder.
- Loss-values, clients, hyperparameters are all logged and stored by pickling.

## Visualization
Some plotting functions are available for:
- Loss-iteration plots.
- Model output comparison plots.
- R^2 value comparison bar graphs.

## Running experiments
Experiments can be run by:
- Setting an `underlying_model` within `src/utils/generate_data.py`
- Setting hyperparameters in `src/hyperparams.py`
- Executing `src/main.py`