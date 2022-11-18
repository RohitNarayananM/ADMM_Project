import numpy as np
from numpy import linalg as la
from scipy import sparse
from numpy.linalg import norm
import time as t
from History import History

def objective(A, b, lamda, x, z):
    a = np.dot(A, x) - b
    p = 1/2*((la.norm(a, 2))**2) + lamda*(la.norm(z, 2)**2)
    return p

def ridge(A, b, lamda, rho, alpha):
    MAX_ITER = 100
    ABSTOL = 10**(-4)
    RELTOL = 10**(-2)

    m, n = A.shape
    # save a matrix-vector multiply
    Atb = A.T*b
    # ADMM solver
    x = np.zeros([n, 1])
    z = np.zeros([n, 1])
    u = np.zeros([n, 1])

    h = {}
    h['objval']     = np.zeros(MAX_ITER)
    h['r_norm']     = np.zeros(MAX_ITER)
    h['s_norm']     = np.zeros(MAX_ITER)
    h['eps_pri']    = np.zeros(MAX_ITER)
    h['eps_dual']   = np.zeros(MAX_ITER)
    h['x'] = np.zeros(MAX_ITER)

    # cache the factorization
    L, U = factor(A, rho)
    #decomposing the (AT.A - rho*I) into L,U 

    history = History()
    for k in range(0, MAX_ITER):
        # x-update
        start_time = t.time()
        q = Atb + rho*(z) - u  # temporary value
        #as x - update is xk = (AT.A - rho*I)^(-1)*(AT*b+rho*z-u)
        if m >= n:
            x = la.solve(U.todense(), la.solve(L.todense(), q))
        else:
            x = q/rho - np.dot(A.T, la.solve(U.todense(), la.solve(L.todense(), np.dot(A, q))))/rho**2
        ##print(x,z)
        # z-update
        zold = z
        x_hat = x #alpha*x + (1 - alpha)*zold
        z = (rho*x_hat + u)/(2*lamda + rho)
        #print(z,zold,lamda)
        #print(z)
        # u-update
        u = u + rho*(x_hat - z)
        time= t.time() - start_time
        # diagnostics, reporting, termination checks
        history.addObjval(objective(A, b, lamda, x, z))
        h['objval'][k] = objective(A, b, lamda, x, z)
        history.addR_norm(la.norm(x - z))
        h['r_norm'][k]   = norm(x-z)
        history.addS_norm(la.norm(-rho*(z-zold)))
        h['s_norm'][k]   = norm(-rho*(z-zold))
        history.addEps_pri(np.sqrt(n)*ABSTOL + RELTOL*np.maximum(la.norm(x), la.norm(-z)))
        h['eps_pri'][k]  = np.sqrt(n)*ABSTOL+ RELTOL*np.maximum(norm(x),norm(-z))
        history.addEps_dual(np.sqrt(n)*ABSTOL + RELTOL*la.norm(rho*u))
        h['eps_dual'][k] = np.sqrt(n)*ABSTOL+ RELTOL*norm(rho*u)
        history.addtime_iter(time)

        if history.getR_norm()[k] < history.getEps_pri()[k] and history.getS_norm()[k]<history.getEps_dual()[k]:
            break
    time = list(history.gettime_iter())
    time_iter = time.copy()
    time_iter.sort(reverse=True)
    history.settime_iter(time_iter)
    return x.ravel(), history ,h  

def factor(A, rho):
    m, n = A.shape
    if m >= n:
        L = la.cholesky(np.dot(A.T, A) + rho*sparse.eye(n))
    else:
        L = la.cholesky(sparse.eye(m) + 1/rho*(np.dot(A, A.T)))

    L = sparse.coo_matrix(L)
    U = sparse.coo_matrix(L.T)

    return L, U