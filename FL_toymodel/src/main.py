#%%
from copy import deepcopy
from datetime import datetime
from typing import Callable, Union

import dill as pickle
import numpy as np
from FL_toymodel.src.client import PolyLRClient
from FL_toymodel.src.experiments.overlapping_segments import init_hyperparam_generator
from FL_toymodel.src.utils.generate_clients import (
    initialize_centralized_client, initialize_clients)
from FL_toymodel.src.utils.generate_data import (generate_data,
                                                 generate_polynomial_weights,
                                                 sum_underlying_components)
from FL_toymodel.src.utils.training import (fedavg_training, fit_linear,
                                            general_training)
from FL_toymodel.src.utils.visualization import (plot_data,
                                                 plot_loss_vs_iteration,
                                                 plot_model_output,
                                                 plot_r2_scores)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import gen_batches


#%%
def initialize_data(
        n_samples: int = 1000,
        input_range: tuple[Union[int, float], Union[int, float]] = (-10, 10),
        noise_std: float = 1e0,
        poly_degree_data: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable]:
    """
    Generates noisy data based on some underlying model, which itself is simply 
    a sum of components (polynomial, sine wave).
    """
    theta_actual = generate_polynomial_weights(input_range[0], input_range[1], poly_degree_data)

    underlying_model = sum_underlying_components(
        lambda x: PolynomialFeatures(degree=poly_degree_data).fit_transform(x[:,np.newaxis]) @ theta_actual,
    )

    x, y, y_noisy = generate_data(input_range, n_samples, underlying_model, noise_std=noise_std)

    return x, y, y_noisy, underlying_model


def initialize_clients_main(
        x: np.ndarray,
        y_noisy: np.ndarray,
        idx_generator: Callable = lambda x: gen_batches(x.shape[0], 10),
        poly_degree_linmodels: int = 1,
        sgd_kwargs: dict[str, Union[int, float]] = {},
        sgd_centralized_kwargs: dict[str, Union[int, float]] = {}
    ) -> tuple[list[PolyLRClient], PolyLRClient, PolyLRClient, PolyLRClient]:
    """
    Initializes several types of clients to be trained.
    """
    sgd_clients = initialize_clients(x, y_noisy, poly_degree_linmodels, idx_generator, **sgd_kwargs)
    sgd_client_combined = initialize_centralized_client(sgd_clients, **sgd_centralized_kwargs)
    sgd_client_centralized = deepcopy(sgd_client_combined)
    lr_client_centralized = deepcopy(sgd_client_combined)
    return sgd_clients, sgd_client_combined, sgd_client_centralized, lr_client_centralized


def train_main(
        sgd_clients: list[PolyLRClient],
        sgd_client_combined: PolyLRClient,
        sgd_client_centralized: PolyLRClient,
        lr_client_centralized: PolyLRClient,
        n_outer_epochs: int = 500
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Train several different clients and obtain intermittent loss-value results.
    """
    # FL using SGD
    sgd_client_combined, sgd_clients, sgd_client_combined_results, sgd_clients_results = fedavg_training(
        sgd_client_combined, sgd_clients, n_outer_epochs=n_outer_epochs)
    # Centralized training using SGD
    sgd_client_centralized, sgd_client_centralized_results = general_training(
        [sgd_client_centralized], n_outer_epochs=n_outer_epochs)
    sgd_client_centralized = sgd_client_centralized[0]
    # Centralized training using linear regression
    lr_client_centralized, lr_client_centralized_results = fit_linear(lr_client_centralized)
    return sgd_clients_results, sgd_client_combined_results, sgd_client_centralized_results, lr_client_centralized_results
    

#%%
def main(hyperparams):
    np.random.seed(hyperparams["random_seed"])

    t_start = datetime.now()

    x, y, y_noisy, underlying_model = initialize_data(
        **hyperparams["data"]
    )

    (   sgd_clients,
        sgd_client_combined,
        sgd_client_centralized,
        lr_client_centralized
    ) = initialize_clients_main(
        x,
        y_noisy,
        **hyperparams["clients"],
    )

    (
        sgd_clients_results,
        sgd_client_combined_results,
        sgd_client_centralized_results,
        lr_client_centralized_results
    ) = train_main(
        sgd_clients,
        sgd_client_combined,
        sgd_client_centralized,
        lr_client_centralized,
        **hyperparams["training"],
    )

    ## Store results
    results = {
        "hyperparams": hyperparams,
        "x": x,
        "y": y,
        "y_noisy": y_noisy,
        "underlying_model": underlying_model,
        "sgd_clients": sgd_clients,
        "sgd_client_combined": sgd_client_combined,
        "sgd_client_centralized": sgd_client_centralized,
        "lr_client_centralized": lr_client_centralized,
        "sgd_clients_results": sgd_clients_results,
        "sgd_client_combined_results": sgd_client_combined_results,
        "sgd_client_centralized_results": sgd_client_centralized_results,
        "lr_client_centralized_results": lr_client_centralized_results,
        "duration": datetime.now() - t_start
    }

    with open(hyperparams["results_save_path"], 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plots
    fig_data, ax_data = plot_data(x, y, y_noisy, n_samples_plot=100)

    fig_loss, ax_loss = plot_loss_vs_iteration(
        **{
            "SGD server": sgd_client_combined_results,
            "SGD clients": sgd_clients_results,
            "SGD centralized": sgd_client_centralized_results,
            "LR centralized": lr_client_centralized_results,
        }
    )

    fig_output, ax_output = plot_model_output(
        x,
        y_noisy,
        n_samples_plot=100,
        **{
            "Underlying model": lambda x: underlying_model(x),
            "SGD server": lambda x: sgd_client_combined.predict(x),
            "SGD centralized": lambda x: sgd_client_centralized.predict(x),
            "LR centralized": lambda x: lr_client_centralized.predict(x),
        }
    )

    fig_output, ax_output = plot_r2_scores(
        x,
        y_noisy,
        **{
            "Underlying model": lambda x: underlying_model(x),
            "SGD server": lambda x: sgd_client_combined.predict(x),
            "SGD centralized": lambda x: sgd_client_centralized.predict(x),
            "LR centralized": lambda x: lr_client_centralized.predict(x),
        }
    )

    duration = results["duration"]
    print(f"Duration: {duration}")

    return


# %%
if __name__ == "__main__":
    hyperparam_generator = init_hyperparam_generator()
    for i, initialize_hyperparam in enumerate(hyperparam_generator):
        hyperparams_i = initialize_hyperparam()
        experiment_name = hyperparams_i["experiment_name"]
        print(f"Starting experiment: {experiment_name} #{i+1}...")
        main(hyperparams_i)
        print(f"Finished experiment: {experiment_name} #{i+1}.")


# %%
