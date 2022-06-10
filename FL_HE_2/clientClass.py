import torch
from sklearn.model_selection import train_test_split
import copy

def client_train_test_split(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
        X_train = torch.tensor(X_train.to_numpy()).float()
        X_test = torch.tensor(X_test.to_numpy()).float()
        y_train = torch.Tensor(y_train.reset_index(drop=True).to_numpy())
        y_test = torch.Tensor(y_test.reset_index(drop=True).to_numpy())
        return X_train, X_test, y_train, y_test
class Client:
    """


    """

    def __init__(self, name, X, y, model: torch.nn.Module, lr, criterion):
        self.name = name
        self.X, self.y, self.lr = X, y, lr
        self.model = model
        self.criterion = criterion
        self.optim = torch.optim.SGD(model.parameters(), lr=lr)
        self.X_train, self.X_test, self.y_train, self.y_test = client_train_test_split(self.X, self.y)

    
    def set_pubkey(self, pubkey):
        self.pubkey = pubkey
    
    def set_model(self, model):
        self.model = model
    
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

