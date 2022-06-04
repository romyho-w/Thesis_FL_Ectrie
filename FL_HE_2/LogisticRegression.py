# Define class for logistic regression with stochastic gradient descent

class LogisticRegression:
    def __init__(self, start, lr=0.01):
        self.lr = lr
        self.beta = start
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, p, y):
        return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
    
    def fit(self, X, y):      
        z = np.dot(X, self.beta)
        p = self.__sigmoid(z)
        gradient = np.dot(X.T, (p - y)) / y.size
        self.beta -= self.lr * gradient
        loss = self.__loss(p, y)
        
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.beta))
    
    def predict_loss(self, X, y):   
        p = self.__sigmoid(np.dot(X, self.beta))
        return self.__loss(p, y)
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    
    def beta(self):
        return self.beta
        
    
def log_loss_NB(naive_bayes_model, X, y):
    p = naive_bayes.predict_proba(chunk_test)[:,1]
    return (-y * np.log(p) - (1 - y) * np.log(1 - p)).sum()