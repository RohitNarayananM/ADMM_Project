import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from numpy.linalg import inv


class Ridge:
    def __init__(self, regularizer_coef=75):
        self.regularizer_coef = regularizer_coef

    def fit(self, train_X, train_y):
        assert isinstance(train_X, (np.ndarray)) == True
        assert isinstance(train_y, (np.ndarray)) == True
        train_X = np.insert(train_X, 0, np.ones((train_X.shape[0])), axis=1)
        self.train_X = train_X
        self.train_y = train_y
        self.N, self.D = train_X.shape
        self.estimate_coef()

    def estimate_coef(self):
        betas = np.matmul(np.matmul(inv(np.matmul(self.train_X.T, self.train_X) + self.regularizer_coef*np.eye(self.D)), self.train_X.T), self.train_y)
        self.coef = betas[1:]
        self.intercept = betas[0]

    def predict(self,test_X,test_y):
        predict_y = np.matmul(test_X,self.coef)
        print('Implemented R2 score: ',r2_score(test_y,predict_y))
        print('ScikitLearn MAE: ',mean_absolute_error(test_y,predict_y))
        print('ScikitLearn MSE: ',mean_squared_error(test_y,predict_y))
