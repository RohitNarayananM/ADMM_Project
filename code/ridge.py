import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score
from numpy.linalg import inv
import matplotlib.pyplot as plt


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

    def predict(self,test_X,test_y,classification=False):
        predict_y = np.matmul(test_X,self.coef)+self.intercept
        if classification:
            predict_y=predict_y > 0.5
            print("Accuracy: ",accuracy_score(test_y,predict_y))
        else:
            print('Implemented R2 score: ',r2_score(test_y,predict_y))
            print('ScikitLearn MAE: ',mean_absolute_error(test_y,predict_y))
            print('ScikitLearn MSE: ',mean_squared_error(test_y,predict_y))
    def coef_values(self, save_name=None):
        coef_dict = {}
        for idx, coef_val in enumerate(self.coef):
            coef_dict[str(idx+1)] = coef_val
        coef_dicts = dict(sorted(coef_dict.items(), key=lambda item: item[1]))
        plt.bar(range(len(coef_dicts)), list(
            coef_dicts.values()), align='center')
        plt.title('Coefficient Values Sorted')
        if save_name:
            plt.savefig(f'fig/{save_name}.png', dpi=300)
        plt.show()
