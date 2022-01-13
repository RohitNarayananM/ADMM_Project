from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train
from ridge import Ridge
from ridge_admm import Ridge as RidgeADMM
import numpy as np
import time

Ridge = Ridge(75)

print("Heart Patient dataset".center(50, "="))
heart_X_train = np.array(heart_X_train)
heart_Y_train = np.array(heart_Y_train)
t=time.time()
Ridge.fit(heart_X_train, heart_Y_train)
print(f"Time :{time.time()-t}")
Ridge.predict(heart_X_test,heart_Y_test)
# Ridge.coef_values()
print("="*50)

A = heart_X_train
b = heart_Y_train
b = b.reshape(b.shape[0], 1)
admm = RidgeADMM(A, b, False)
t = time.time()
for i in range(0, 20):
    admm.step()
print(f"Time :{time.time()-t}")
admm.predict(heart_X_test, heart_Y_test)
print("="*50)

student_X_train = np.array(student_X_test)
student_Y_train = np.array(student_Y_test)
t=time.time()
Ridge.fit(student_X_train, student_Y_train)
print(f"Time :{time.time()-t}")
Ridge.predict(student_X_test,student_Y_test)
# Ridge.coef_values()
print("="*50)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
admm=RidgeADMM(A, b,False)
t=time.time()
for i in range(0, 20):
    admm.step()
print(f"Time :{time.time()-t}")
admm.predict(student_X_test,student_Y_test)