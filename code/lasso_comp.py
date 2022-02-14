from preprocessing import student_X_test, student_X_train, student_Y_test, student_Y_train, pollution_X_test, pollution_X_train, pollution_Y_test, pollution_Y_train
from lasso import Lasso
from lasso_admm import Lasso as LassoADMM
import time

Lasso = Lasso()
PARALLEL=True

print("Pollution dataset".center(50, "="))
t=time.time()
Lasso.fit(pollution_X_train, pollution_Y_train)
print(f"Time :{time.time()-t}")
Lasso.predict(pollution_X_test,pollution_Y_test)
print("="*50)

A = pollution_X_train
b = pollution_Y_train
b=b.reshape(b.shape[0],1)
admm = LassoADMM(A, b,PARALLEL)
arr1=[]
obj=admm.LassoObjective()
for i in range(0, 50):
    t = time.time()
    admm.step()
    arr1.append((time.time()-t)*1000)
    if(admm.LassoObjective()-obj<0.001):
        break
print(f"Time :{sum(arr1)/1000}")
admm.predict(pollution_X_test,pollution_Y_test)
print("Student Performance dataset".center(50, "="))
t=time.time()
Lasso.fit(student_X_train, student_Y_train)
print(f"Time :{time.time()-t}")
Lasso.predict(student_X_test, student_Y_test)
print("="*50)

A=student_X_train
b=student_Y_train
b=b.reshape(b.shape[0],1)
admm = LassoADMM(A, b,PARALLEL)
arr2=[]
obj=admm.LassoObjective()
for i in range(0, 50):
    t = time.time()
    admm.step()
    arr2.append((time.time()-t)*1000)
    if(admm.LassoObjective()-obj<0.001):
        break
print(f"Time :",sum(arr2)/1000)
admm.predict(student_X_test,student_Y_test)

# plt.plot(arr1)
# plt.plot(arr2)
# plt.plot(arr1, ls="", marker="o")
# plt.plot(arr2, ls="", marker="o")
# plt.title("Time of each iteration Lasso")
# plt.xlabel("Iteration")
# plt.ylabel("Time in milliseconds")
# plt.legend(["Heart Patient dataset", "Student Performance dataset"])
# plt.show()