from preprocessing import student_X_test, student_X_train, student_Y_test, student_Y_train, pollution_X_test, pollution_X_train, pollution_Y_test, pollution_Y_train
from ridge import Ridge
from ridge_admm import Ridge as RidgeADMM
import time

Ridge = Ridge(75)
PARALLEL=True

print("Pollution dataset".center(50, "="))
t=time.time()
Ridge.fit(pollution_X_train, pollution_Y_train)
print(f"Time :{time.time()-t}")
Ridge.predict(pollution_X_test,pollution_Y_test)
print("="*50)

A = pollution_X_train
b = pollution_Y_train
b = b.reshape(b.shape[0], 1)
admm = RidgeADMM(A, b, PARALLEL)
arr1=[]
obj=admm.RidgeObjective()
for i in range(0, 50):
    t = time.time()
    admm.step()
    if(abs(obj-admm.RidgeObjective())<0.001):
        break
    arr1.append((time.time()-t)*1000)
print("{i}Time :",sum(arr1)/1000)
admm.predict(pollution_X_test, pollution_Y_test)

print("Student Performance dataset".center(50, "="))
t=time.time()
Ridge.fit(student_X_train, student_Y_train)
print(f"Time :{time.time()-t}")
Ridge.predict(student_X_test,student_Y_test)
print("="*50)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
admm=RidgeADMM(A, b,PARALLEL)
arr2=[]
obj=admm.RidgeObjective()
for i in range(0, 50):
    t = time.time()
    admm.step()
    if(abs(obj-admm.RidgeObjective())<0.001):
        break
    arr2.append((time.time()-t)*1000)
print("Time :",sum(arr2)/1000)
admm.predict(student_X_test,student_Y_test)