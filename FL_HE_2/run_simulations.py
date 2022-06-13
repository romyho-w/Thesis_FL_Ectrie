import copy
import json
import random
import time
from datetime import datetime
from typing import Generator

import diffprivlib.models as dp
import numpy as np
import tenseal as ts
import tensorflow as tf

from dataFunction import *
from FL_utils import *
from HE_functions import *
from lrClass import LR
from make_logreg_data import *
from simulationDataUtils import *

random.seed(11007303)
np.random.seed(2021)
def run_simulations(N_Clients, N_Features, N_Observations, monte_carlo_reps, mean_distance):
    EPOCHS = 80
    poly_mod_degree = 4096
    coeff_mod_bit_sizes = [40, 20, 40]

    # create TenSEALContext
    ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)

    # scale of ciphertext to use
    ctx_eval.global_scale = 2 ** 20

    # this key is needed for doing dot-product operations
    ctx_eval.generate_galois_keys()

    iter = 0
    Total_dict = {}
    for n_clients in N_Clients:
        for n_features in N_Features:
            for mean_dist in mean_distance:
                glob_model = LR(n_features)
                for n_observations in N_Observations:
                    parameters = {}
                    parameters['n_clients'] = n_clients
                    parameters['n_features'] = n_features
                    parameters['n_observations'] = n_observations
                    parameters['mean_dist'] = mean_dist
                    
                
                    KL_overview = []
                    best_acc_overview = []
                    KL_dict = {}
                    # for mu in mu_options:
                    # glob_model = LR(n_features)
                    # print(glob_model.state_dict())
                    for mc in monte_carlo_reps:
                        client_distribution_list = make_clients_dist(mean_dist, n_clients, n_features)
                        KL_df, KL_sym_df = make_KL_matrices(n_clients, client_distribution_list)
                        clients = define_clients(client_distribution_list, n_observations,n_features,glob_model)
                        validation_X_set, validation_y_set = make_validation_sets(clients)
                        fl_glob_model = copy.deepcopy(glob_model)
                        best_epoch, best_acc, model_dict, final_results = FL_proces(clients, validation_X_set, validation_y_set, ctx_eval, fl_glob_model, iters= 100)
                        KL_mean = np.array(KL_sym_df)[np.triu_indices(n_clients, k=1)].mean()
                        print('inter:{}, mean dist: {}, n_observation:{}, n_clients:{}, n_features:{}, mc:{}'.format(iter, mean_dist, n_observations, n_clients,  n_features, mc))
                        print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))  
                        # print(model_dict)
                        # save_results.append([KL_sym_df[0][1],best_acc])
                        KL_overview.append(KL_mean)
                        best_acc_overview.append(float(best_acc))
                    # KL_dict['KL'] = KL_overview
                    # KL_dict['acc'] = best_acc_overview
                    KL_dict['KL_mean'] = np.mean(KL_overview)
                    KL_dict['acc_mean'] = np.mean(best_acc_overview)
                    
                summary = {}
                summary['parameters'] = parameters
                summary['results'] = KL_dict
                Total_dict[iter] = summary
                iter += 1
                        
                            # print(final_results)
    return Total_dict


if __name__ == "__main__":

    # N_Clients = [2, 4]
    # N_Features = [10]
    # N_Observations = [100]
    # monte_carlo_reps = range(2)
    # mean_distance = [0.1]
    # n_option = 10
    # cov_range = [1, 10]

    # N_Clients = [2,5,10,30]
    # N_Features = [2,5,10,15,20]
    # N_Observations = [100,200,500]
    # mean_distance = np.arange(0.1, 3.0, 0.1)
    # monte_carlo_reps = range(100)
    

    N_Clients = [2,5,10,30]
    N_Features = [2,5,10,15,20]
    N_Observations = [200]
    mean_distance = np.arange(0.1, 3.0, 0.5)
    monte_carlo_reps = range(40)

    results = run_simulations(N_Clients, N_Features, N_Observations, monte_carlo_reps, mean_distance)
    time_str = str(datetime.now())
    with open('total_dict_'+time_str+'.json', 'w') as convert_file:
     convert_file.write(json.dumps(results))
