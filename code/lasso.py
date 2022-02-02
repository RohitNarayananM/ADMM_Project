import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score
import matplotlib.pyplot as plt
import collections

class Lasso:
    def __init__(self, regularizer_coef=50):
        self.regularizer_coef = regularizer_coef

    def calculate_ro_j(self, current_j):
        other_column_indices = [i for i in range(self.D) if i != current_j]
        coefs_without_j = self.betas[other_column_indices]
        X_without_j = self.train_X[:, other_column_indices]
        ro = np.sum(np.multiply(self.train_y-np.sum(np.multiply(coefs_without_j,X_without_j), axis=1), self.train_X[:, current_j]), axis=0)
        return ro

    def fit(self,train_X,train_y):
        assert isinstance(train_X,(np.ndarray)) == True
        assert isinstance(train_y,(np.ndarray)) == True
        train_X = np.insert(train_X,0,np.ones((train_X.shape[0])),axis=1)
        self.train_X = train_X
        self.train_y = train_y
        self.N, self.D = train_X.shape
        self.estimate_coef()

    def estimate_coef(self, epoch=50):
        self.betas = np.zeros(self.D)
        for e in range(epoch):
            for d in range(self.D):
                ro = self.calculate_ro_j(d)
                denominator = np.sum(self.train_X[:, d]**2, axis=0)
                if ro > self.regularizer_coef:
                    self.betas[d] = (-self.regularizer_coef+ro) / denominator
                elif ro < -self.regularizer_coef:
                    self.betas[d] = (self.regularizer_coef+ro) / denominator
                else:
                    self.betas[d] = 0
        self.coef = self.betas[1:]
        self.intercept = self.betas[0]

    def predict(self,test_X,test_y,classification=False):
        predict_y = np.matmul(test_X,self.coef)+self.intercept
        if classification:
            predict_y=predict_y > 0.5
            print("Accuracy: ",accuracy_score(test_y,predict_y))
        else:
            print('Implemented R2 score: ',r2_score(test_y,predict_y))
            print('ScikitLearn MAE: ',mean_absolute_error(test_y,predict_y))
            print('ScikitLearn MSE: ',mean_squared_error(test_y,predict_y))