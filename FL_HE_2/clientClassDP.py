import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import log_loss


def client_train_test_split(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

        X_train = torch.tensor(X_train.to_numpy()).float()
        X_test = torch.tensor(X_test.to_numpy()).float()
        y_train = torch.Tensor(y_train.reset_index(drop=True).to_numpy()).unsqueeze(1)
        y_test = torch.Tensor(y_test.reset_index(drop=True).to_numpy()).unsqueeze(1)
        return X_train, X_test, y_train, y_test


def compute_loss(y_true, y_pred):
    # binary cross entropy
    y_zero_loss = y_true * np.log(y_pred + 1e-9)
    y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
    return -np.mean(y_zero_loss + y_one_loss)

class Client:
    """


    """

    def __init__(self, name, X, y, cat_feat, model):
        self.name = name
        self.X, self.y, self.cat_feat = X, y, cat_feat
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = client_train_test_split(self.X, self.y)

    
    def set_pubkey(self, pubkey):
        self.pubkey = pubkey
    
    def set_model(self, model):
        self.model = model

    def train(self, n_epochs=10): 
        epoch_loss = []
        for e in range(n_epochs):
            self.model.fit(self.X_train, np.ravel(self.y_train))
            acc = self.model.score(self.X_test, np.ravel(self.y_test))
            y_pred = self.model.predict(self.X_test)
            loss = compute_loss(np.ravel(self.y_test), y_pred)
            epoch_loss.append(loss)
        
        return self.model, epoch_loss
    
    # def set_state_dict(self, state_dict):
    #     self.model.load_state_dict(state_dict)

