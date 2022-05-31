import builtins

import numpy as np
import tenseal.sealapi as seal
import torch
from cryptotree.cryptotree import (HomomorphicNeuralRandomForest,
                                   HomomorphicTreeEvaluator,
                                   HomomorphicTreeFeaturizer)
from cryptotree.polynomials import polyeval_tree
from cryptotree.preprocessing import Featurizer
from cryptotree.seal_helper import (append_globals_to_builtins,
                                    create_seal_globals)
from cryptotree.tree import (CrossEntropyLabelSmoothing, NeuralRandomForest,
                             SigmoidTreeMaker, TanhTreeMaker)
from fastai.basic_data import DataBunch
from fastai.metrics import accuracy
from fastai.tabular.learner import Learner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch.utils import data


class TabularDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X: np.ndarray, y: np.ndarray):
        'Initialization'
        self.X, self.y = X,y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = torch.tensor(self.X[index]).float()
        y = torch.tensor(self.y[index])

        return X, y

def split_prep_data(client, pipe):
    # X_train, X_test, y_train, y_test = train_test_split(
    #                                                     client.X,
    #                                                     client.y,
    #                                                     test_size=0.2,
    #                                                     random_state=42)
    X_train_normalized, X_valid_normalized, y_train, y_valid = train_test_split(pipe.fit_transform(client.X),
                                                        client.y,
                                                        test_size=0.2,
                                                        random_state=42)
    # y_train = y_train.astype(int)
    # y_valid = y_valid.astype(int)
    x_train = torch.tensor(X_train_normalized).float()
    y_train = torch.Tensor(y_train.reset_index(drop=True)).unsqueeze(1)
    x_test = torch.tensor(X_valid_normalized).float()
    y_test = torch.Tensor(y_valid.reset_index(drop=True)).unsqueeze(1)
    return y_train, y_test, x_train, x_test


def make_tabularDataset(X_train_normalized, y_train, X_valid_normalized, y_valid):
    train_ds = TabularDataset(X_train_normalized, y_train.values)
    valid_ds = TabularDataset(X_valid_normalized, y_valid.values)
    
    return train_ds, valid_ds
    
def make_dataloader(train_ds, valid_ds, bs=128):    
    # bs = 128

    train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = data.DataLoader(valid_ds, batch_size=bs)
    fix_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=False)
    return train_dl, valid_dl, fix_dl


def set_tree(max_depth, dilatation_factor):
    max_depth = max_depth
    polynomial_degree = dilatation_factor
    sigmoid_tree_maker = SigmoidTreeMaker(
                                        use_polynomial=True,
                                        dilatation_factor=dilatation_factor,
                                        polynomial_degree=polynomial_degree)

    tanh_tree_maker = TanhTreeMaker(
                                    use_polynomial=True,
                                    dilatation_factor=dilatation_factor,
                                    polynomial_degree=polynomial_degree)

                                    
    rf = RandomForestClassifier(n_estimators=20, random_state=2,max_depth=max_depth, bootstrap=False)
    # rf = RandomForestClassifier(max_depth=max_depth, random_state=1, bootstrap=False)

    return rf, sigmoid_tree_maker, tanh_tree_maker 


def fine_tune_tree(tree_maker, rf, train_dl, valid_dl, fix_dl):
    model = NeuralRandomForest(rf.estimators_, tree_maker=tree_maker)

    model.freeze_layer("comparator")
    model.freeze_layer("matcher")

    data = DataBunch(train_dl, valid_dl,fix_dl=fix_dl)

    criterion = CrossEntropyLabelSmoothing()

    learn = Learner(data, model, loss_func=criterion, metrics=accuracy)
    learn.lr_find()
    learn.recorder.plot()

    learn.fit_one_cycle(5,1e-1/2)

    return model

def set_config():
    dilatation_factor = 16
    degree = dilatation_factor

    PRECISION_BITS = 28
    UPPER_BITS = 9

    polynomial_multiplications = int(np.ceil(np.log2(degree))) + 1
    n_polynomials = 2
    matrix_multiplications = 3

    depth = matrix_multiplications + polynomial_multiplications * n_polynomials

    poly_modulus_degree = 32768

    moduli = [PRECISION_BITS + UPPER_BITS] + (depth) * [PRECISION_BITS] + [PRECISION_BITS + UPPER_BITS]

    create_seal_globals(globals(), poly_modulus_degree, moduli, PRECISION_BITS, use_symmetric_key=False)
    append_globals_to_builtins(globals(), builtins)



def setup_HE(tree_maker, model):

    h_rf = HomomorphicNeuralRandomForest(model)
    tree_evaluator = HomomorphicTreeEvaluator.from_model(h_rf, tree_maker.coeffs, 
                                                   polyeval_tree, evaluator, encoder, relin_keys, galois_keys, 
                                                   scale)

    homomorphic_featurizer = HomomorphicTreeFeaturizer(h_rf.return_comparator(), encoder, encryptor, scale)

    return tree_evaluator, homomorphic_featurizer


def predict(x, homomorphic_featurizer, tree_evaluator):
    """Performs HRF prediction"""
    
    # We first encrypt and evaluate our model on it
    ctx = homomorphic_featurizer.encrypt(x)
    outputs = tree_evaluator(ctx)
    
    # We then decrypt it and get the first 2 values which are the classes scores
    ptx = seal.Plaintext()
    decryptor.decrypt(outputs, ptx)
    
    homomorphic_pred = encoder.decode_double(ptx)[:2]
    homomorphic_pred = np.argmax(homomorphic_pred)
    
    return homomorphic_pred


def HE_RF_pred(X_valid_normalized, homomorphic_featurizer, tree_evaluator):

    print("Homomorpic encryption prediction calculations ....")
    hrf_pred = []
    # row_num = 1
    for i in X_valid_normalized:
        # print("row "+ str(row_num)+ " of the "+ str(len(X_valid_normalized)) )
        hrf_pred.append(predict(i, homomorphic_featurizer, tree_evaluator))
        # row_num += 1
    return hrf_pred

def NRF_pred(valid_dl, model):
    print("Neural tree prediction calculations ....")
    outputs = []

    for x,y in valid_dl:
        with torch.no_grad():
            pred = model(x)
        outputs.append(pred)
        
    nrf_pred = torch.cat(outputs).argmax(dim=1).numpy()

    return nrf_pred

def LIN_pred(X_train_normalized, y_train, X_valid_normalized):
    print("Linear prediction calculations ....")
    linear = LogisticRegression()
    linear.fit(X_train_normalized, y_train)

    # We compute the linear preds
    linear_pred = linear.predict(X_valid_normalized)

    return linear_pred


def compute_metrics(pred, y):
    """Computes all the metrics between predictions and real values"""
    accuracy = accuracy_score(pred,y)
    precision = precision_score(pred,y)
    recall = recall_score(pred,y)
    f1 = f1_score(pred, y)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)

def results_trees(rf, sigmoid_tree_maker, tanh_tree_maker, X_train_normalized, y_train ):    
    rf.fit(X_train_normalized, y_train)
    
    estimators = rf.estimators_
    sigmoid_neural_rf = NeuralRandomForest(estimators, sigmoid_tree_maker)
    tanh_neural_rf = NeuralRandomForest(estimators, tanh_tree_maker)

    with torch.no_grad():
        sigmoid_neural_pred = sigmoid_neural_rf(torch.tensor(X_train_normalized).float()).argmax(dim=1).numpy()
        tanh_neural_pred = tanh_neural_rf(torch.tensor(X_train_normalized).float()).argmax(dim=1).numpy()

    pred = rf.predict(X_train_normalized)
    print(f"Original accuracy : {(pred == y_train).mean()}")

    print(f"Accuracy of sigmoid  : {(sigmoid_neural_pred == y_train).mean()}")
    print(f"Accuracy of tanh : {(tanh_neural_pred == y_train).mean()}")

    print(f"Match between sigmoid and original : {(sigmoid_neural_pred == pred).mean()}")
    print(f"Match between tanh and original : {(tanh_neural_pred == pred).mean()}")
    return rf, sigmoid_neural_rf, tanh_neural_rf