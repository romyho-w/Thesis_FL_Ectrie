#%%
from FL_toymodel.src.utils.general import update_nested_dict
import csv
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Generator

from FL_toymodel.src.config import (CHANGING_HP_FILENAME, DEFAULT_HYPERPARAMS,
                                    EXPERIMENT_DIR_BASE)

EXPERIMENT_NAME = "inner_epochs"
EXPERIMENT_DIR = os.path.join(EXPERIMENT_DIR_BASE, EXPERIMENT_NAME)
CHANGING_HP_FILEPATH = os.path.join(EXPERIMENT_DIR, CHANGING_HP_FILENAME)


#%%
def init_hyperparam_generator() -> Generator[Callable, None, None]:
    if not os.path.isdir(EXPERIMENT_DIR): os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    with open(CHANGING_HP_FILEPATH, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["experiment_name", "inner_epochs", "random_seed"])

    for inner_epochs in (1, 5, 10, 50):
        for random_seed in range(10):
            def init_hyperparams() -> dict[str, Any]:
                current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                experiment_filename = f"{EXPERIMENT_NAME}_{current_datetime}.pickle"
                
                with open(CHANGING_HP_FILEPATH, "a") as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=",")
                    csvwriter.writerow([experiment_filename, inner_epochs, random_seed])
                
                hyperparams = deepcopy(DEFAULT_HYPERPARAMS)
                hyperparams_update = {
                    "experiment_name": EXPERIMENT_NAME,
                    "random_seed": random_seed,
                    "results_save_path": os.path.join(EXPERIMENT_DIR, experiment_filename),
                    "clients": {
                        "sgd_kwargs": {"inner_epochs": inner_epochs},
                    },
                }

                update_nested_dict(hyperparams, hyperparams_update)

                return hyperparams
            
            yield init_hyperparams


#%%
