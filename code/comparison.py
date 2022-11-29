from preprocessing import fraud_X_test,fraud_X_train,fraud_Y_test,fraud_Y_train
from test_lasso import compare_lasso
from test_ridge import compare_ridge
from matplotlib import pyplot as plt

print("Lasso Comparison".center(100,"="))
print("Fraud Dataset".center(50,"="))
lasso_x,lasso_y,lasso_time=compare_lasso(fraud_X_train,fraud_X_test,fraud_Y_train,fraud_Y_test)

print("Ridge Comparison".center(100,"="))
print("Fraud Dataset".center(50,"="))
ridge_x,ridge_y,ridge_time=compare_ridge(fraud_X_train,fraud_X_test,fraud_Y_train,fraud_Y_test)

plt.plot(lasso_x, lasso_time,label="Lasso Gradient Descent")
plt.xlabel('iter (k)')
plt.ylabel('time for each iteration')
plt.title("Lasso VS Ridge Gradient Descent")
plt.plot(ridge_x, ridge_time,label="Ridge Gradient Descent")
plt.legend()
plt.savefig('../images/lasso_VS_Ridge_GD_time_for_iter.png')
plt.show()

plt.plot(lasso_x,lasso_y,label="Lasso ADMM")
plt.xlabel('iter (k)')
plt.ylabel('time for each iteration')
plt.title("lasso Vs Ridge ADMM")
plt.plot(ridge_x, ridge_y,label="Ridge ADMM")
plt.legend()
plt.savefig('../images/lasso_VS_Ridge_ADMM_time_for_iter.png')
plt.show()