#%%
from FL_toymodel.src.experiments.inner_epochs import CHANGING_HP_FILEPATH, EXPERIMENT_DIR
import csv
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Generator

from FL_toymodel.src.config import (CHANGING_HP_FILENAME, DEFAULT_HYPERPARAMS,
                                    EXPERIMENT_DIR_BASE)
from FL_toymodel.src.utils.general import update_nested_dict
from FL_toymodel.src.utils.generate_clients import uniform_overlapping_segments

EXPERIMENT_NAME = "overlapping_segments"
EXPERIMENT_DIR = os.path.join(EXPERIMENT_DIR_BASE, EXPERIMENT_NAME)
CHANGING_HP_FILEPATH = os.path.join(EXPERIMENT_DIR, CHANGING_HP_FILENAME)


#%%
def init_hyperparam_generator() -> Generator[Callable, None, None]:
    if not os.path.isdir(EXPERIMENT_DIR): os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    with open(CHANGING_HP_FILEPATH, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["experiment_name", "overlap", "random_seed"])

    for overlap in (0.0, 0.1, 0.5, 0.8):
        for random_seed in range(10):
            def init_hyperparams() -> dict[str, Any]:
                current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                experiment_filename = f"{EXPERIMENT_NAME}_{current_datetime}.pickle"
                
                with open(CHANGING_HP_FILEPATH, "a") as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=",")
                    csvwriter.writerow([experiment_filename, overlap, random_seed])
                
                hyperparams = deepcopy(DEFAULT_HYPERPARAMS)
                hyperparams_update = {
                    "experiment_name": EXPERIMENT_NAME,
                    "random_seed": random_seed,
                    "results_save_path": os.path.join(EXPERIMENT_DIR, experiment_filename),
                    "clients": {
                        "idx_generator": lambda x: uniform_overlapping_segments(x, 40, overlap)(),
                    },
                }

                update_nested_dict(hyperparams, hyperparams_update)

                return hyperparams
            
            yield init_hyperparams


#%%
