from preprocessing import heart_X_test, heart_X_train, heart_Y_test, heart_Y_train, student_X_test, student_X_train, student_Y_test, student_Y_train
from lasso import Lasso
from lasso_admm import Lasso as LassoADMM
import time

Lasso = Lasso()
PARALLEL=False

print("Heart Patient dataset".center(50, "="))
t=time.time()
Lasso.fit(heart_X_train, heart_Y_train)
print(f"Time :{time.time()-t}")
Lasso.predict(heart_X_test,heart_Y_test,True)
print("="*50)

A = heart_X_train
b = heart_Y_train
b=b.reshape(b.shape[0],1)
admm = LassoADMM(A, b,PARALLEL)
arr1=[]
for i in range(0, 50):
    t = time.time()
    admm.step()
    arr1.append((time.time()-t)*1000)
print(f"Time :{sum(arr1)/1000}")
admm.predict(heart_X_test,heart_Y_test,True)

print("Student Performance dataset".center(50, "="))
t=time.time()
Lasso.fit(student_X_train, student_Y_train)
print(f"Time :{time.time()-t}")
Lasso.predict(student_X_test, student_Y_test,False)
print("="*50)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
admm = LassoADMM(A, b,PARALLEL)
arr2=[]
for i in range(0, 50):
    t = time.time()
    admm.step()
    arr2.append((time.time()-t)*1000)
print("Time :",sum(arr2)/1000)
admm.predict(student_X_test,student_Y_test,False)

# plt.plot(arr1)
# plt.plot(arr2)
# plt.plot(arr1, ls="", marker="o")
# plt.plot(arr2, ls="", marker="o")
# plt.title("Time of each iteration Lasso")
# plt.xlabel("Iteration")
# plt.ylabel("Time in milliseconds")
# plt.legend(["Heart Patient dataset", "Student Performance dataset"])
# plt.show()