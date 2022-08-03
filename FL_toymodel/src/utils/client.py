#%%
from typing import Union

import numpy as np
from FL_toymodel.src.client import PolyLRClient
from sklearn.linear_model import LinearRegression


#%%
def aggregate_jacobians(clients: list[PolyLRClient]) -> np.ndarray:
    """
    Aggregate the Jacobians of a list of clients by averaging them.
    """
    n_clients, n_weights = len(clients), clients[0]._linear_regressor.coef_.shape[0]
    aggregated_jacobian = sum((client.jacobian_train() for client in clients), start=np.zeros(n_weights))
    aggregated_jacobian *= (1/n_clients)
    return aggregated_jacobian


def get_average_lincoefs(clients: list[PolyLRClient]) -> np.ndarray:
    """
    Calculates a weighted average (by number of training samples) 
    of the linear model weights of a list of clients.
    """
    n_samples = np.array([client._n_train for client in clients])
    client_lincoefs = [client.get_weights()[:,np.newaxis] for client in clients]
    client_lincoefs = np.concatenate(client_lincoefs, axis=1)
    average_lincoef = (client_lincoefs @ n_samples) / n_samples.sum()
    return average_lincoef


def aggregate_linmodel_from_clients(clients: list[PolyLRClient]) -> LinearRegression:
    """
    Aggregates the linear models of a list of clients into a linear model.
    """
    average_lincoef = get_average_lincoefs(clients)
    return linmodel_from_lincoef(average_lincoef)


def linmodel_from_lincoef(lincoef: np.ndarray) -> LinearRegression:
    """
    Creates a linear model from a vector with model weights.
    """
    linear_regressor = LinearRegression(fit_intercept=False)
    linear_regressor.coef_ = lincoef
    linear_regressor.intercept_ = 0.
    return linear_regressor


def aggregate_client_data(clients: list[PolyLRClient], seed: Union[None, int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate the training and testing data of a list of clients into a total combined data set
    and shuffles it before returning.    
    """
    # Concatenate training data
    client_x_train = np.concatenate([client._x_train for client in clients], axis=0)
    client_y_train = np.concatenate([client._y_train for client in clients], axis=0)
    # Concatenate testing data
    client_x_test = np.concatenate([client._x_test for client in clients], axis=0)
    client_y_test = np.concatenate([client._y_test for client in clients], axis=0)
    # Combine training and testing data if available
    if client_y_test.shape[0] > 0:
        client_x = np.concatenate([client_x_train, client_x_test], axis=0)
        client_y = np.concatenate([client_y_train, client_y_test], axis=0)
    else:
        client_x = client_x_train
        client_y = client_y_train
    # Shuffle total data
    idx_xy = np.arange(client_x.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx_xy)
    client_x, client_y = client_x[idx_xy], client_y[idx_xy]
    return client_x, client_y


#%%
