import random 
import numpy as np
import pandas as pd
import copy
from clientClass import Client
import torch

def make_clients_dist(mean_dist, n_clients, n_features):
    clients_distribution = []
    random.seed(11007303)
    np.random.seed(2021)
    for i in range(n_clients):
        mu = np.random.uniform(-mean_dist, mean_dist, n_features)
        cov = np.diag([1]*n_features)
        clients_distribution.append([mu, cov])
    return clients_distribution



def make_labels(X: np.ndarray, thresholds, epsilon_sigma) -> np.ndarray:
    thresholds = np.array(thresholds)[:, None]
    epsilon = np.random.default_rng().normal(0, epsilon_sigma, X.shape[0])[:,None]
    return (X @ thresholds + epsilon) > 0

def define_clients(clients_distribution, n_observations,n_features, glob_model, epsilon_sigma):
    clients = []
    weights = np.random.default_rng().normal( 0, 1, size=n_features)
    for i in clients_distribution:
        samples = np.random.default_rng().multivariate_normal(i[0], i[1], n_observations)
        x = pd.DataFrame(samples)
        
        y = make_labels(x, weights, epsilon_sigma ).replace({True: 1, False: 0})
        lr = 0.2
        client_model = copy.deepcopy(glob_model)
        clients.append(Client(i, x, y, client_model, lr,  torch.nn.BCELoss(reduction='mean'), weights) )
    return clients


def make_validation_sets_hypercubes(clients, n_features, samples, epsilon_sigma):
    frames = [ f.X for f in clients ]
    all_X = pd.concat(frames)

    min_X_values = []
    max_X_values = []
    for i in all_X.columns:
        min_X_values.append( all_X.describe()[i]['min'])
        max_X_values.append( all_X.describe()[i]['max'])
    
    val_x = []
    for i in range(n_features):
        val_x.append(torch.tensor(np.random.default_rng().uniform(min_X_values[i], max_X_values[i], size= samples)))

    validation_X_set = torch.stack(val_x, -1).float()
    weights = clients[0].weights
    validation_y_set = make_labels(validation_X_set, weights, epsilon_sigma).float()

    return validation_X_set, validation_y_set
