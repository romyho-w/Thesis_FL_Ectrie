import copy
import json
import random
from datetime import datetime

import numpy as np
import tenseal as ts

from dataFunction import *
from HE_functions import *
from lrClass import LR
from simulationDataUtils import *

random.seed(11007303)
np.random.seed(2021)
def run_simulations(N_Clients, N_Features, N_Observations, monte_carlo_reps, mean_distance, epsilon_sigmas):
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
                    for epsilon_sigma in epsilon_sigmas:
                        parameters = {}
                        parameters['n_clients'] = n_clients
                        parameters['n_features'] = n_features
                        parameters['n_observations'] = n_observations
                        parameters['mean_dist'] = mean_dist
                        parameters['epsilon_sigma'] = epsilon_sigma
                        
                    
                        KL_overview = []
                        KL_y_overview = []
                        best_acc_overview = []
                        KL_dict = {}
                        # CI_overview = []

                        for mc in monte_carlo_reps:
                            client_distribution_list = make_clients_dist(mean_dist, n_clients, n_features)
                            clients = define_clients(client_distribution_list, n_observations,n_features,glob_model, epsilon_sigma)
                            KL_df = make_KL_matrice_simulation(n_clients, client_distribution_list)
                            KL_y = make_KL_matrices_y(n_clients, clients)
                            validation_X_set, validation_y_set = make_validation_sets_hypercubes(clients, n_features, 100, epsilon_sigma)
                            
                            fl_glob_model = copy.deepcopy(glob_model)
                            best_epoch, best_acc, model_dict, final_results, client_results = FL_proces(clients, validation_X_set, validation_y_set, ctx_eval, fl_glob_model, 100, True, False)
                            KL_mean = (np.array(KL_df)[np.triu_indices(n_clients, k=1)].mean() + np.array(KL_df)[np.tril_indices(n_clients, -1)].mean()) /2

                            KL_y_mean  = (np.array(KL_y)[np.triu_indices(n_clients, k=1)].mean() + np.array(KL_y)[np.tril_indices(n_clients, -1)].mean()) /2

                            print('inter:{}, mean dist: {}, n_observation:{}, n_clients:{}, n_features:{}, mc:{}, epsilon_sigma:{}'.format(iter, mean_dist, n_observations, n_clients,  n_features, mc, epsilon_sigma))
                            print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))  
                            print('N_clients: {}, KL_y: {}'.format(n_clients, KL_y_mean))

                            KL_overview.append(KL_mean)
                            KL_y_overview.append(KL_y_mean)
                            best_acc_overview.append(float(best_acc))

                        KL_dict['acc_min'] = np.min(best_acc_overview)
                        KL_dict['acc_max'] = np.max(best_acc_overview)
                        KL_dict['KL_mean'] = np.mean(KL_overview)
                        KL_dict['KL_y_mean'] = np.mean(KL_y_overview)
                        KL_dict['acc_mean'] = np.mean(best_acc_overview)

                        summary = {}
                        summary['parameters'] = parameters
                        summary['results'] = KL_dict
                        Total_dict[iter] = summary
                        iter += 1
                                

    return Total_dict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
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
    # epsilon_sigmas = [0.5, 1, 10, 100, 1000]



    # N_Clients = [2,5,10,30, 40, 100]
    # N_Features = [15]
    # N_Observations = [200]
    # mean_distance = np.arange(0.1, 3.0, 0.5)
    # monte_carlo_reps = range(100)
    # epsilon_sigmas = [0.5, 1, 10, 100]
    # client_list_begin = np.array([2, 5])
    # N_Clients = np.concatenate((client_list_begin, np.arange(10, 150, 10)))

# N_Clients
    N_Clients = [2]
    N_Features = [5]
    N_Observations = [2]
    mean_distance = [1.1]
    monte_carlo_reps = range(1)
    epsilon_sigmas = [10]
    

    # N_Clients = [2,5,10,30]
    # N_Features = [2,5,10,15,20]
    # N_Observations = [200]
    # mean_distance = np.arange(0.1, 3.0, 0.5)
    # monte_carlo_reps = range(40)

    # N_Clients = [5]
    # N_Features = [15]
    # N_Observations = [200]
    # # mean_distance = np.arange(0.1, 3.0, 0.5)
    # mean_distance = [1.5]
    # monte_carlo_reps = range(1)
    # epsilon_sigmas = [0.5, 1, 10, 100, 1000]

    results = run_simulations(N_Clients, N_Features, N_Observations, monte_carlo_reps, mean_distance, epsilon_sigmas)
    time_str = str(datetime.now())
    with open('Results/total_dict_'+time_str+'.json', 'w') as convert_file:
     convert_file.write(json.dumps(results, cls=NpEncoder))
