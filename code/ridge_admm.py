import numpy as np
from os import getpid
from numpy.linalg import inv
from numpy.linalg import norm
from multiprocessing import Process, cpu_count
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,accuracy_score


class SolveIndividual:
    def solve(self, A, b, nu, rho, Z):
        t1 = A.dot(A.T)
        A = A.reshape(-1, 1)
        tX = (A * b + rho * Z - nu) / (t1 + rho)
        return tX

class CombineSolution:
    def combine(self, nuBar, xBar, Z, rho):
        t = nuBar.reshape(-1, 1)
        t = t + rho * (xBar.reshape(-1, 1) - Z)
        return t.T


class Ridge:
    def __init__(self, A, b, parallel=False):
        self.D = A.shape[1]
        self.N = A.shape[0]
        if parallel:
            self.XBar = np.zeros((self.N, self.D))
            self.nuBar = np.zeros((self.N, self.D))
        self.nu = np.zeros((self.D, 1))
        self.rho = 1
        self.X = np.random.randn(self.D, 1)
        self.Z = np.zeros((self.D, 1))
        self.A = A
        self.b = b
        self.alpha = 0.01
        self.parallel = parallel
        self.numberOfThreads = 32

    def step(self):
        if self.parallel:
            return self.step_parallel()

        self.X = inv(self.A.T.dot(self.A) + self.rho).dot(self.A.T.dot(self.b) + self.rho * self.Z - self.nu)

        self.Z = self.rho* self.X + self.nu / (2*self.alpha + self.rho)
        self.nu = self.nu + self.rho * (self.X - self.Z)

    def solveIndividual(self, i):
        solve = SolveIndividual()
        t = solve.solve(self.A[i], np.asscalar(self.b[i]), self.nuBar[i].reshape(-1, 1), self.rho, self.Z)
        return t.T

    def combineSolution(self, i):
        combine = CombineSolution()
        return combine.combine(self.nuBar[i].reshape(-1, 1), self.XBar[i].reshape(-1, 1), self.Z, self.rho)

    def step_parallel(self):

        temp=Parallel(n_jobs = self.numberOfThreads, backend = "threading")(delayed(self.solveIndividual)(i) for i in range(0, self.N-1))

        self.X = np.average(temp, axis=0)
        self.X = self.X.reshape(-1, 1)
        self.nu = self.nu.reshape(-1, 1)
        self.Z = self.rho* self.X + self.nu / (2*self.alpha + self.rho)

        temp=Parallel(n_jobs = self.numberOfThreads, backend = "threading")(delayed(self.combineSolution)(i) for i in range(0, self.N-1))
        self.nu=np.average(temp,axis=0)

    def step_iterative(self):
        temp=self.XBar
        for i in range(0, self.N-1):
            self.solveIndividual(i,temp)

        self.X = np.average(temp, axis=0)
        self.X = self.X.reshape(-1, 1)
        self.nu = self.nu.reshape(-1, 1)
        self.Z = self.rho* self.X + self.nu / (2*self.alpha + self.rho)

        temp=self.nuBar
        for i in range(0, self.N-1):
            self.combineSolution(i,temp)
        self.nu=np.average(temp,axis=0)

    def RidgeObjective(self):
        return 0.5 * norm(self.A.dot(self.X) - self.b)**2 + self.alpha * norm(self.X, 1)

    def predict(self,test_X,test_y,classification=False):
        predict_y = np.matmul(test_X,self.X)
        if classification:
            predict_y=predict_y > 0.5
        print('Implemented R2 score: ',r2_score(test_y,predict_y))
        print('ScikitLearn MAE: ',mean_absolute_error(test_y,predict_y))
        print('ScikitLearn MSE: ',mean_squared_error(test_y,predict_y))
