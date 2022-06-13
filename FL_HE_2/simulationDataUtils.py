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
    random_mu = np.random.randint(-10,10,n_features)
    for i in range(n_clients):
        if i == 0: 
            mu = random_mu
        else:
            mu = np.random.uniform(random_mu, (random_mu * (1+mean_dist)))
        cov = np.diag([1]*n_features)
        clients_distribution.append([mu, cov])
    return clients_distribution


def KL_divergence(dist1, dist2):
    mu1 = dist1[0]
    cov1 = dist1[1]

    mu2 = dist2[0]
    cov2 = dist2[1]

    mu_dif = mu2 - mu1
    inv_cov2 = np.linalg.inv(cov2)
    trace_cov12 = np.matrix.trace(inv_cov2*cov1)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)

    return 1/2 *( mu_dif.T @ inv_cov2 @ mu_dif+trace_cov12-np.log(det_cov1/det_cov2)-len(mu1))


def make_KL_matrices(n_clients, clients_distribution):
    kl = np.empty((n_clients, n_clients))
    kl_sym = np.empty((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            kl[i,j] = KL_divergence(clients_distribution[i], clients_distribution[j])
            kl_sym[i,j] = (KL_divergence(clients_distribution[i], clients_distribution[j]) +KL_divergence(clients_distribution[j], clients_distribution[i]))/2
    KL_df = pd.DataFrame(kl)
    KL_sym_df = pd.DataFrame(kl_sym)
    return KL_df, KL_sym_df

def make_labels(X: np.ndarray, thresholds) -> np.ndarray:
    thresholds = np.array(thresholds)[:, None]
    return (X @ thresholds) > 0

def define_clients(clients_distribution, n_observations,n_features, glob_model):
    clients = []
    for i in clients_distribution:
        samples = np.random.default_rng().multivariate_normal(i[0], i[1], n_observations)
        x = pd.DataFrame(samples)
        weights = np.random.default_rng().integers(low=-10, high=10, size=n_features)
        y = make_labels(x, weights ).replace({True: 1, False: 0})
        lr = 0.2
        client_model = copy.deepcopy(glob_model)
        clients.append(Client(i, x, y, client_model, lr,  torch.nn.BCELoss(reduction='mean' )) )
    return clients

def make_validation_sets(clients):
    validation_X_set = torch.tensor(())
    validation_y_set = torch.tensor(())
    for i in range(len(clients)):
        validation_X_set = torch.cat((validation_X_set, clients[i].X_test), 0)
        validation_y_set = torch.cat((validation_y_set, clients[i].y_test), 0)
    return validation_X_set, validation_y_set