from lasso_comp import arr1, arr2
print("Ridge".center(50, "="))
from ridge_comp import arr1 as arr3
from ridge_comp import arr2 as arr4
import matplotlib.pyplot as plt

plt.plot(arr1)
plt.plot(arr2)
plt.plot(arr3)
plt.plot(arr4)
plt.plot(arr1, ls="", marker="o")
plt.plot(arr2, ls="", marker="o")
plt.plot(arr3, ls="", marker="o")
plt.plot(arr4, ls="", marker="o")
plt.title("Time of each iteration")
plt.xlabel("Iteration")
plt.ylabel("Time in milliseconds")
plt.legend(["Heart Patient dataset Lasso", "Student Performance dataset Lasso", "Heart Patient dataset Ridge", "Student Performance dataset Ridge"])
plt.show()