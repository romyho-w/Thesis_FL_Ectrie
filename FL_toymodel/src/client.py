#%%
from __future__ import annotations

from copy import deepcopy
from typing import Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import gen_batches


#%%
class PolyLRClient(object):
    def __init__(
        self: PolyLRClient,
        polynomial_features: PolynomialFeatures,
        logistic_regressor: LogisticRegression,
        x: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        test_size: Union[float, None] = 0.25,
        batch_size: Union[float, None] = None,
        inner_epochs: int = 1,
        learning_rate: float = 1e-3,
        random_state: Union[None, int] = None,
        
        ):

        self._polynomial_features = deepcopy(polynomial_features)
        # self._logistic_regressor = deepcopy(logistic_regressor)
        self.intercept = np.ones((x.shape[0], 1))
        self.x = np.concatenate((self.intercept, x), axis=1)
        if test_size is not None:
            x_train, x_test, y_train, y_test = train_test_split(self.x, y, test_size=test_size, random_state=random_state)
        else:
            x_train, x_test, y_train, y_test = x.copy(), np.array([]), y.copy(), np.array([])

        self._x_train = x_train
        self._y_train = y_train
        self._n_train = self._y_train.shape[0]
        self._x_test = x_test
        self._y_test = y_test
        self.set_weights(beta)
        self._n_test = self._y_test.shape[0]
        
        self._polynomial_features.fit(self._x_train[:,np.newaxis])

        self._batch_size = batch_size if batch_size is not None else max(1, self._y_train.shape[0] // 10)
        self._inner_epochs = inner_epochs
        self._learning_rate = learning_rate

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, p, y):
        # Binary cross entropy loss
        return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
    
    def fit(self, X, y):
        beta = self.get_weights()      
        z = np.dot(X, beta)
        p = self.__sigmoid(z)
        gradient = np.dot(X.T, (p - y)) / y.size
        self._beta -= self._learning_rate * gradient
        # loss = self.__loss(p, y)
        
    def predict_prob(self, X):
        beta = self.get_weights()  
        return self.__sigmoid(np.dot(X, beta))
    
    def predict_loss(self, X, y):   
        beta = self.get_weights()  
        p = self.__sigmoid(np.dot(X, beta))
        return self.__loss(p, y)
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
        
    
    def loss_train(self: PolyLRClient) -> float:
        """
        Calculates the loss value of the client's model based on its training data.
        """
        return self.__loss(self.predict_prob(self._x_train), self._y_train)

    def loss_test(self: PolyLRClient) -> float:
        """
        Calculates the loss value of the client's model based on its testing data.
        """
        return self.__loss(self.predict_prob(self._x_test), self._y_test)


    def do_inner_iterations(self: PolyLRClient) -> PolyLRClient:
        """
        Perform a client training iteration, consisting of self._inner_epochs inner epochs.
        """
        for inner_epoch in range(self._inner_epochs):
            for idx_batch in gen_batches(self._n_train, self._batch_size):
                self.fit(self._x_train[idx_batch], self._y_train[idx_batch])
        return self

    def set_weights(self: PolyLRClient, new_weights: np.ndarray) -> PolyLRClient:
        """
        Set the weights of the client's model.
        """
        self._beta = new_weights.copy()
        return self

    def get_weights(self: PolyLRClient) -> np.ndarray:
        """
        Get the weights of the client's model.
        """
        return self._beta


#%%
