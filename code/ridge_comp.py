from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train
from ridge import Ridge
from ridge_admm import Ridge as RidgeADMM
import time

Ridge = Ridge(75)
PARALLEL=True

print("Heart Patient dataset".center(50, "="))
t=time.time()
Ridge.fit(heart_X_train, heart_Y_train)
print(f"Time :{time.time()-t}")
Ridge.predict(heart_X_test,heart_Y_test,True)
print("="*50)

A = heart_X_train
b = heart_Y_train
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
print("Time :",sum(arr1)/1000)
admm.predict(heart_X_test, heart_Y_test,True)

print("Student Performance dataset".center(50, "="))
t=time.time()
Ridge.fit(student_X_train, student_Y_train)
print(f"Time :{time.time()-t}")
Ridge.predict(student_X_test,student_Y_test,False)
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