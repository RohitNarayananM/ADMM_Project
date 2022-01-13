import numpy as np
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train

def predict(test_X,test_y,print_score=True):
    global X
    predict_y = np.matmul(test_X,X)
    if print_score:
        print('Implemented R2 score: ',r2_score(test_y,predict_y))
        print('ScikitLearn MAE: ',mean_absolute_error(test_y,predict_y))
        print('ScikitLearn MSE: ',mean_squared_error(test_y,predict_y))
    return r2_score(test_y,predict_y)


num_iterations = 20

A = heart_X_train
b = heart_Y_train
b=b.reshape(b.shape[0],1)
print('Heart Disease prediction')
print(A.shape, b.shape)

admm = optim.ADMM([A,b,0.01],"lasso")
print(admm.getLoss())
for i in range(0, num_iterations):
    print('O Val:', admm.step())
X = admm.getWeights()
admm.predict(heart_X_test,heart_Y_test)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
print('Student performance prediction')
print(A.shape, b.shape)

admm = optim.ADMM([A, b, 0.01], "lasso")
print(admm.getLoss())
for i in range(0, num_iterations):
    print('O Val:', admm.step())
X = admm.getWeights()
admm.predict(student_X_test,student_Y_test)

A=heart_X_train
b=heart_Y_train
b=b.reshape(b.shape[0],1)
print(A.shape, b.shape)

admm = optim.ADMM([A, b, 0.01], "ridge")
print(admm.getLoss())
for i in range(0, num_iterations):
    print('O Val:', admm.step())
X = admm.getWeights()
admm.predict(heart_X_test,heart_Y_test)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
print(A.shape, b.shape)

admm = optim.ADMM([A, b, 0.01], "ridge")
print(admm.getLoss())
for i in range(0, num_iterations):
    print('O Val:', admm.step())
