from preprocessing import fraud_X_test,fraud_X_train,fraud_Y_test,fraud_Y_train
from test_lasso import compare_lasso
from test_ridge import compare_ridge

print("Lasso Comparison".center(100,"="))
print("Fraud Dataset".center(50,"="))
compare_lasso(fraud_X_train,fraud_X_test,fraud_Y_train,fraud_Y_test)

print("Ridge Comparison".center(100,"="))
print("Fraud Dataset".center(50,"="))
compare_ridge(fraud_X_train,fraud_X_test,fraud_Y_train,fraud_Y_test)