#%%
from copy import deepcopy

import numpy as np
from FL_toymodel.src.client import PolyLRClient
from FL_toymodel.src.utils.client import get_average_lincoefs


#%%
def fedavg_training(
    server: PolyLRClient,
    clients: list[PolyLRClient],
    n_outer_epochs: int = 1000,
    info_interval: int = 100
    ) -> tuple[PolyLRClient, list[PolyLRClient], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Performs federated averaging training with a list of clients and logs loss-values.
    """

    n_clients = len(clients)
    print(f"Starting federated averaging training with {n_clients} clients...")

    # Pre-allocate loss arrays
    _nan_server = np.full((n_outer_epochs+1, 1), np.nan)
    _nan_clients = np.full((n_outer_epochs+1, n_clients), np.nan)
    server_results = {"train": _nan_server.copy(), "test": _nan_server.copy()}
    clients_results = {"train": _nan_clients.copy(), "test": _nan_clients.copy()}

    # Initial global model broadcast
    for client in clients:
        client.set_weights(server.get_weights())

    # Log initial loss values
    server_results["train"][0,0] = server.loss_train()
    server_results["test"][0,0] = server.loss_test()
    clients_results["train"][0,:] = np.array([client.loss_train() for client in clients])
    clients_results["test"][0,:] = np.array([client.loss_test() for client in clients])

    # Training loop
    for outer_epoch in range(1, n_outer_epochs+1):
        if (outer_epoch) % info_interval == 0: print(f"\tOuter epoch #{outer_epoch}/{n_outer_epochs}")

        # Perform inner training iterations
        for client in clients:
            client.do_training_iteration()

        # Push models to orchestrator and aggregate into global model
        server.set_weights(get_average_lincoefs(clients))

        # Broadcast new global model to clients
        for client in clients:
            client.set_weights(server.get_weights())

        # Log results for this epoch
        server_results["train"][outer_epoch,0] = server.loss_train()
        server_results["test"][outer_epoch,0] = server.loss_test()
        clients_results["train"][outer_epoch,:] = np.array([client.loss_train() for client in clients])
        clients_results["test"][outer_epoch,:] = np.array([client.loss_test() for client in clients])

    print("Finished.")

    clients_results_average = average_result(clients, clients_results)

    return server, clients, server_results, clients_results_average


def average_result(clients: list[PolyLRClient], results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Calculates a weighted average of loss-values that have been gathered for a list of clients.
    """
    result_average = deepcopy(results)
    n_training_samples = np.array([client._n_train for client in clients])
    weighting = n_training_samples / n_training_samples.sum()
    for kind in ("train", "test"):
        result_average[kind] = (result_average[kind] @ weighting)[:,np.newaxis]
    return result_average


def general_training(
    clients: list[PolyLRClient],
    n_outer_epochs: int = 1000, info_interval: int = 100
    ) -> tuple[list[PolyLRClient], dict[str, np.ndarray]]:
    """
    Performs iterative training of a list of clients and logs results.
    """
    
    n_clients = len(clients)
    print(f"Starting SGD training with {n_clients} clients...")

    # Pre-allocate loss arrays
    _nan_clients = np.full((n_outer_epochs+1, n_clients), np.nan)
    results = {"train": _nan_clients.copy(), "test": _nan_clients.copy()}

     # Log results for this epoch
    results["train"][0,:] = np.array([client.loss_train() for client in clients])
    results["test"][0,:] = np.array([client.loss_test() for client in clients])

    # Train loop
    for outer_epoch in range(1, n_outer_epochs+1):
        for client in clients:
            if (outer_epoch) % info_interval == 0: print(f"\tOuter epoch #{outer_epoch}/{n_outer_epochs}")
            client.do_training_iteration()
        # Log results for this epoch
        results["train"][outer_epoch,:] = np.array([client.loss_train() for client in clients])
        results["test"][outer_epoch,:] = np.array([client.loss_test() for client in clients])
    
    print("Finished.")

    results_average = average_result(clients, results)

    return clients, results_average


def fedavg_fit_linear(
    server: PolyLRClient,
    clients: list[PolyLRClient],
    ) -> tuple[PolyLRClient, list[PolyLRClient], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Trains a set of clients by solving the linear regression problem in one step analytically, 
    and aggregates the result into the server. Logs loss-values after training. 
    """

    # Train loop
    for client in clients:
        client.fit_linear()

    server.set_weights(get_average_lincoefs(clients))
    
    # Log results
    server_results = {
        "train": np.array([[server.loss_train()]]),
        "test": np.array([[server.loss_test()]]),
    }

    client_results = {
        "train": np.array([[client.loss_train() for client in clients]]),
        "test": np.array([[client.loss_test() for client in clients]])
    }

    client_results_average = average_result(clients, client_results)
    
    return server, clients, server_results, client_results_average


def fit_linear(
    client: PolyLRClient,
    ) -> tuple[PolyLRClient, dict[str, np.ndarray]]:
    """
    Trains a client by linear regression. Logs loss-values after training. 
    """
    client.fit_linear()

    # Log results
    results = {
        "train": np.array([[client.loss_train()]]),
        "test": np.array([[client.loss_test()]]),
    }

    return client, results


# %%
