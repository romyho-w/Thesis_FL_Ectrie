import copy 
import tenseal as ts
import numpy as np
import pandas as pd
import torch
from clientClass import Client
from lrClass import LR

def accuracy_loss_LR(model, x, y, simulation=False):
    if simulation:
        criterion = torch.nn.BCELoss()
        out = model(x)
        correct = torch.abs(y - out) < 0.5
        loss = criterion(out, y)
    else:
        criterion = torch.nn.BCELoss()
        out = model(x)
        yhat = torch.squeeze(out)

        correct = torch.abs(y - yhat) < 0.5
        loss = criterion(yhat, y)
    return correct.float().mean(), loss.item()

def average_state_dict(state_dicts):
    result = copy.deepcopy(state_dicts[0])
    for key in result:
        for state_dict in state_dicts[1:]:
            result[key] += state_dict[key]
        result[key] *= 1/len(state_dicts)
    return result

def encrypt_state_dicts(state_dicts, ctx_eval):
    for loc, state_dict in enumerate(state_dicts):
        for key in state_dict.keys():
            if key == 'lr.weight':
                var_list = state_dict.get(key)[0].tolist()
                encrypted_values = ts.ckks_vector(ctx_eval, var_list)
                state_dict[key] = encrypted_values
            else:
                var_list = state_dict.get(key).tolist()
                encrypted_bias = ts.CKKSTensor(ctx_eval,var_list)
                state_dict[key] = encrypted_bias
        state_dicts[loc] = state_dict
    return state_dicts

def decrypt_state_dicts(state_dict):
    for key in state_dict.keys():
        if key == 'lr.weight':
            state_dict[key] = torch.Tensor(state_dict[key].decrypt()).unsqueeze(0)
        else:
            state_dict[key] = torch.Tensor(state_dict[key].decrypt().tolist())
    return state_dict

def FL_proces(clients, validation_X_set, validation_y_set, ctx_eval, glob_model, iters, simulation= False, switzerland=False):
    loss_train = []
    net_best = None
    best_acc = None
    best_epoch = None
    results = []
    min_loss_client = []
    if switzerland:
        client_results = {'Cleveland': [],
        'Switzerland': [],
        'VA Long Beach': [],
        'Hungary': []}
    else:
        client_results = {'Cleveland': [],
        'VA Long Beach': [],
        'Hungary': []}
    glob_model.eval()
    enrypted_state_dicts= None
    acc_test, loss_test =  accuracy_loss_LR(glob_model,validation_X_set, validation_y_set, simulation)

    best_acc = acc_test
    for iter in range(iters):
        loss_locals = []
        client_state_dicts = []
        client_acc_list = []
        client_loss_list = []
        for client in clients:
            client_model = copy.deepcopy(glob_model)
            client.set_state_dict(client_model.state_dict())
            client_state_dict, loss = train_model_client(client, epochs=40, simulation = simulation)
            
            loss_locals.append(copy.deepcopy(loss))
            min_loss_client.append(min(loss))
            client_acc, client_loss = accuracy_loss_LR(client.model, client.X_test, client.y_test)
            new_list = client_results.get(client.name)
            new_list.append(client_acc)
            client_results[client.name] = new_list
            client_state_dicts.append(client_state_dict)

        enrypted_state_dicts = encrypt_state_dicts(copy.deepcopy(client_state_dicts), ctx_eval)
        averaged_encrypted_state_dict = average_state_dict(enrypted_state_dicts)
        decrypted_state_dicts = decrypt_state_dicts(averaged_encrypted_state_dict)
        glob_model.load_state_dict(decrypted_state_dicts)

        loss_avg = sum(min_loss_client) / len(min_loss_client)
        loss_train.append(loss_avg)        
            
        acc_test, loss_test =  accuracy_loss_LR(glob_model,validation_X_set, validation_y_set, simulation)


        if best_acc is None or acc_test > best_acc:
            net_best = copy.deepcopy(glob_model)
            best_acc = acc_test
            best_epoch = iter

        results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
        final_results = np.array(results)
        final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
    
    return best_epoch, best_acc, glob_model.state_dict(), final_results, client_results

def train_model_client(client:Client, epochs, simulation=False):
    epoch_loss = []

    for e in range(epochs):
        client.model.train()
        client.optim.zero_grad()
        out = client.model(client.X_train)
        if not simulation:
            yhat = torch.squeeze(out)
        else:
            yhat = out
        loss = client.criterion(yhat, client.y_train)
        if e == 0:
            epoch_loss.append(loss.item())
        loss.backward()
        client.optim.step()
        epoch_loss.append(loss.item())
    return client.model.state_dict(), epoch_loss
