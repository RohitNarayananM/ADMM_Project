import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error,r2_score
from lasso_admm import lasso
from lasso import LassoRegression
import warnings
warnings.filterwarnings("ignore")


PLOT = False
ITER = False

def Standard_error1(sample):
    std = np.std(sample,ddof=0) #standard deviation
    standard_error = std/math.sqrt(len(sample))
    return standard_error

def compare_lasso(x_tr,x_te,y_tr,y_te):
    factors = ['target']
    X_train, X_test, y_train, y_test = x_tr, x_te, y_tr, y_te
    A = np.matrix(X_train)
    b = np.matrix(y_train).T
    lambda_max = la.norm(np.dot(A.T, b), np.inf)

    lamda = 0.000001*lambda_max

    x, history, h = lasso(A, b, lamda, 1.0, 1.0)
    y_test_predict=X_test.dot(x[0].T)
    if ITER:
        print('%3s' % 'iter', '%10s' % 'r norm', '%10s' % 'eps pri', '%10s' % 's norm', '%10s' % 'eps dual', '%10s' % 'objective','%8s' % 'time')
        for k in range(len(history.getObjval())):
            print('%3d' % k, '%10.4f' % history.getR_norm()[k], '%10.4f' % history.getEps_pri()[k], '%10.4f' % history.getS_norm()[k],
                        '%10.4f' % history.getEps_dual()[k], '%10.2f' % history.getObjval()[k],'%11.6f' % history.gettime_iter()[k])
    
    coef = []
    for i in range(len(factors)):
        coef.append((factors[i],round(float(x.T[i][0]),4)))

    if PLOT:
        K = len(history.gettime_iter())
        x = np.arange(K)
        plt.plot(x, history.gettime_iter())
        plt.xlabel('iter (k)')
        plt.ylabel('time for each iteration')
        plt.show()

        plt.plot(x, history.getObjval())
        plt.xlabel('iter (k)')
        plt.ylabel('f(x^k) + g(z^k)')
        plt.show()

        plt.subplot(211)
        plt.plot(x, np.maximum(10**(-8), history.getR_norm()), '-', history.getEps_pri(), '--')
        plt.ylabel('||r||_2')

        plt.subplot(212)
        plt.plot(x, np.maximum(10**(-8), history.getS_norm()), '-', history.getEps_dual(), '--')
        plt.xlabel('iter (k)')
        plt.ylabel('||s||_2')
        plt.show()

    # model evaluation (MSE,MAE,std_error)
    mse_predict = mean_squared_error(y_test,y_test_predict)
    r2_predict = r2_score(y_test,y_test_predict)
    std_error = Standard_error1(y_test_predict)

    print ('Std Error is:'+str(std_error))
    print ('MSE is:'+str(mse_predict))
    print ('R2 Score is:'+str(r2_predict))

    run_time = 0
    for k in range(len(history.getObjval())):
        run_time += history.gettime_iter()[k]
    print(round(run_time,5))

    X_train, X_test, y_train, y_test = x_tr, x_te, y_tr, y_te

    model = LassoRegression(iterations = 100, learning_rate = 0.001, l1_penality = 100 )
    model , time = model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    K = len(time)
    x_vals = np.arange(K)
    if PLOT:
        plt.plot(x_vals, time)
        plt.xlabel('iter (k)')
        plt.ylabel('time for each iteration')
        plt.show()

        plt.plot(x_vals, model.objval_grad)
        plt.xlabel('iter (k)')
        plt.ylabel('objective vals')
        plt.show()

    # model evaluation (MSE,MAE,std_error)
    y_pred=np.nan_to_num(y_pred)
    mse_predict = mean_squared_error(y_test,y_pred)
    r2_predict = r2_score(y_test,y_pred)
    std_error = Standard_error1(y_pred)

    print ('Std Error is:'+str(std_error))
    print ('MSE is:'+str(mse_predict))
    print ('R2 Score is:'+str(r2_predict))

    runtime = 0
    for k in range(len(time)):
        runtime += time[k]
    print(round(runtime,5))