import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import collections
from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train

class Lasso:
    def __init__(self, regularizer_coef=50):
        self.regularizer_coef = regularizer_coef
    def calculate_ro_j(self, current_j):
        other_column_indices = [i for i in range(self.D) if i != current_j]
        coefs_without_j = self.betas[other_column_indices]
        X_without_j = self.train_X[:, other_column_indices]
        ro = np.sum(np.multiply(self.train_y-np.sum(np.multiply(coefs_without_j,X_without_j), axis=1), self.train_X[:, current_j]), axis=0)
        return ro
    def count_zero_coefs(self):
        counts = collections.Counter(self.coef)
        print(
            f'There are total of {counts[0]} zeros in coefficients out of {len(self.coef)}')
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
        update_threshold = 10
        #while update_threshold > 5:
        for e in range(epoch):
            for d in range(self.D):
                # Calculate except the bias term
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
    def predict(self,test_X,test_y,print_score=True):
        predict_y = np.matmul(test_X,self.coef)+self.intercept
        if print_score:
            print('Implemented R2 score: ',self.r2_score(test_y,predict_y))
            print('ScikitLearn MAE: ',mean_absolute_error(test_y,predict_y))
            print('ScikitLearn MSE: ',mean_squared_error(test_y,predict_y))
        return r2_score(test_y,predict_y)
    def r2_score(self,y_true,y_pred):
        y_bar = np.mean(y_true)
        SSR = np.sum((y_true-y_pred)**2)
        SST = np.sum((y_true-y_bar)**2)
        r2=  1-(SSR/SST)
        return r2
    def coef_values(self,save_name=None):
        coef_dict={}
        for idx,coef_val in enumerate(self.coef):
            coef_dict[str(idx+1)]= coef_val
        coef_dicts= dict(sorted(coef_dict.items(), key=lambda item: item[1]))
        plt.bar(range(len(coef_dicts)), list(coef_dicts.values()), align='center')
        plt.title('Coefficient Values Sorted')
        if save_name:
            plt.savefig(f'fig/{save_name}.png',dpi=300)
        plt.show()


Lasso = Lasso()

heart_X_train=np.array(heart_X_train)
heart_Y_train=np.array(heart_Y_train)
print(heart_X_train.shape,heart_Y_train.shape,heart_X_test.shape,heart_Y_test.shape)
Lasso.fit(heart_X_train, heart_Y_train)
Lasso.predict(heart_X_test,heart_Y_test)
Lasso.coef_values()

student_X_train=np.array(student_X_train)
student_Y_train=np.array(student_Y_train)
print(student_X_train.shape,student_Y_train.shape,student_X_test.shape,student_Y_test.shape)
Lasso.fit(student_X_train, student_Y_train)
Lasso.predict(student_X_test,student_Y_test)
Lasso.coef_values()
