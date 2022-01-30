from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train
from ridge import Ridge
from ridge_admm import Ridge as RidgeADMM
import numpy as np
import time
import matplotlib.pyplot as plt

<<<<<<< HEAD
Ridge = Ridge()
=======
Ridge = Ridge(75)
PARALLEL=False
>>>>>>> 7b73efe (Added graphs)

print("Heart Patient dataset".center(50, "="))
heart_X_train = np.array(heart_X_train)
heart_Y_train = np.array(heart_Y_train)
t=time.time()
Ridge.fit(heart_X_train, heart_Y_train)
print(f"Time :{time.time()-t}")
<<<<<<< HEAD
Ridge.predict(heart_X_test,heart_Y_test)
Ridge.coef_values()
=======
Ridge.predict(heart_X_test,heart_Y_test,True)
# Ridge.coef_values()
>>>>>>> 7b73efe (Added graphs)
print("="*50)

A = heart_X_train
b = heart_Y_train
b = b.reshape(b.shape[0], 1)
admm = RidgeADMM(A, b,PARALLEL)
arr1=[]
for i in range(0, 20):
    t = time.time()
    admm.step()
    arr1.append((time.time()-t)*1000)
print("Time :",sum(arr1))
admm.predict(heart_X_test, heart_Y_test,True)

print("Student Performance dataset".center(50, "="))

student_X_train = np.array(student_X_test)
student_Y_train = np.array(student_Y_test)
t=time.time()
Ridge.fit(student_X_train, student_Y_train)
print(f"Time :{time.time()-t}")
<<<<<<< HEAD
Ridge.predict(student_X_test,student_Y_test)
Ridge.coef_values()
=======
Ridge.predict(student_X_test,student_Y_test,False)
# Ridge.coef_values()
>>>>>>> 7b73efe (Added graphs)
print("="*50)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
admm=RidgeADMM(A, b,PARALLEL)
arr2=[]
for i in range(0, 20):
    t = time.time()
    admm.step()
    arr2.append((time.time()-t)*1000)
print("Time :",sum(arr2))
admm.predict(student_X_test,student_Y_test,False)

# plt.plot(arr1)
# plt.plot(arr2)
# plt.plot(arr1, ls="", marker="o")
# plt.plot(arr2, ls="", marker="o")
# plt.title("Time of each iteration Ridge")
# plt.xlabel("Iteration")
# plt.ylabel("Time in milliseconds")
# plt.legend(["Heart Patient dataset", "Student Performance dataset"])
# plt.show()