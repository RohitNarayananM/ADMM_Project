from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train
from lasso import Lasso
from lasso_admm import Lasso as LassoADMM
import numpy as np
import time

Lasso = Lasso()

print("Heart Patient dataset".center(50, "="))
t=time.time()
Lasso.fit(heart_X_train, heart_Y_train)
print(f"Time :{time.time()-t}")
Lasso.predict(heart_X_test,heart_Y_test)
# Lasso.coef_values()
print("="*50)

A = heart_X_train
b = heart_Y_train
b=b.reshape(b.shape[0],1)
admm = LassoADMM(A, b,False)
t = time.time()
for i in range(0, 20):
    admm.step()
print(f"Time :{time.time()-t}")
admm.predict(heart_X_test,heart_Y_test)
print("="*50)


print("Student Prformance Dataset".center(50, "="))
t=time.time()
Lasso.fit(student_X_train, student_Y_train)
print(f"Time :{time.time()-t}")
Lasso.predict(student_X_test, student_Y_test)
# Lasso.coef_values()
print("="*50)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
admm = LassoADMM(A, b,False)
t = time.time()
for i in range(0, 20):
    admm.step()
print(f"Time :{time.time()-t}")
admm.predict(student_X_test,student_Y_test)
