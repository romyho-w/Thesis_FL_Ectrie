import torch
import copy
import tenseal as ts


def accuracy_loss_LR(model, x, y):
    criterion = torch.nn.BCELoss()
    out = model(x)
    correct = torch.abs(y - out) < 0.5
    loss = criterion(out, y)
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
