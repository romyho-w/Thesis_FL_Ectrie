#%%
import os

from sklearn.utils import gen_batches


#%%
EXPERIMENT_DIR_BASE = os.path.join("FL_toymodel", "experiment_results")
CHANGING_HP_FILENAME = "changing_hyperparameters.csv"


# %% Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "experiment_name": "default",
    "random_seed": 0,
    "results_save_path": os.path.join(EXPERIMENT_DIR_BASE, "default", "default.pickle"),
    "data": {
        "n_samples": 1_000,
        "input_range": (-10, 10),
        "noise_std": 1e2,
        "poly_degree_data": 3,
    },
    "clients": {
        "idx_generator": lambda x: gen_batches(x.shape[0], 25),
        "poly_degree_linmodels": 3,
        "sgd_kwargs": {"learning_rate": 2e-6, "batch_size": 1, "inner_epochs": 1, "test_size": None},
        "sgd_centralized_kwargs": {"learning_rate": 6e-6, "batch_size": 32},
    },
    "training": {
        "n_outer_epochs": 5_000,
    }
}


#%%