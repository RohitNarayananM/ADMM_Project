import numpy as np
from lasso_admm import Lasso
from ridge_admm import Ridge
from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train


num_iterations = 20

A = heart_X_train
b = heart_Y_train
b=b.reshape(b.shape[0],1)
print(A.shape, b.shape)

admm = Lasso(A, b,False)
print(admm.LassoObjective())

for i in range(0, num_iterations):
    admm.step()
    # print('O Val:', admm.LassoObjective())
admm.predict(heart_X_test,heart_Y_test)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
print(A.shape, b.shape)

admm = Lasso(A, b,False)
print(admm.LassoObjective())
for i in range(0, num_iterations):
    admm.step()
    # print('O Val:', admm.LassoObjective())
admm.predict(student_X_test,student_Y_test)

# A=heart_X_train
# b=heart_Y_train
# b=b.reshape(b.shape[0],1)
# print(A.shape, b.shape)

# admm = Ridge(A, b,False)
# print(admm.LassoObjective())
# for i in range(0, num_iterations):
#     admm.step()
#     # print('O Val:', admm.LassoObjective())
# admm.predict(heart_X_test,heart_Y_test)

# A=student_X_train
# b=student_Y_train
# b=b.reshape(b.shape[0],1)
# print(A.shape, b.shape)

# admm=Ridge(A, b,False)
# print(admm.LassoObjective())
# for i in range(0, num_iterations):
#     admm.step()
#     # print('O Val:', admm.LassoObjective())
# admm.predict(student_X_test,student_Y_test)
