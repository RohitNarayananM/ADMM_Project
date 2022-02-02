import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from joblib import Parallel, delayed
from multiprocessing import Process, Manager, cpu_count, Pool
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score


class SolveIndividual:
    def solve(self, A, b, nu, rho, Z):
        t1 = A.T.dot(A)
        A = A.reshape(-1, 1)
        tX = (A * b + rho * Z - nu) / (t1 + rho)
        return tX


class CombineSolution:
    def combine(self, nuBar, xBar, Z, rho):
        t = nuBar.reshape(-1, 1)
        t = t + rho * (xBar.reshape(-1, 1) - Z)
        return t.T


class Lasso:
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
        self.numberOfThreads = cpu_count()

    def step(self):
        if self.parallel:
            return self.step_iterative()

        # Solve for X_t+1
        self.X = inv(self.A.T.dot(self.A) + self.rho).dot(self.A.T.dot(self.b) + self.rho * self.Z - self.nu)

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - (self.alpha / self.rho) * np.sign(self.Z)
        # Combine
        self.nu = self.nu + self.rho * (self.X - self.Z)

    def solveIndividual(self, i):
        solve = SolveIndividual()
        return solve.solve(self.A[i], np.asscalar(self.b[i]), self.nuBar[i].reshape(-1, 1), self.rho, self.Z)

    def combineSolution(self, i):
        combine = CombineSolution()
        return combine.combine(self.nuBar[i].reshape(-1, 1), self.XBar[i].reshape(-1, 1), self.Z, self.rho)

    def step_parallel(self):
        # Solve for X_t+1
        #Parallel(n_jobs = self.numberOfThreads, backend = "threading")(
        #    delayed(self.solveIndividual)(i) for i in range(0, self.N-1))
        process = []
        for i in range(0, self.N-1):
            p = Process(target=self.solveIndividual, args=(i,))
            p.start()
            process.append(p)

        for p in process:
            p.join()

        self.X = np.average(self.XBar, axis=0)

        self.X = self.X.reshape(-1, 1)
        self.nu = self.nu.reshape(-1, 1)

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - (self.alpha / self.rho) * np.sign(self.Z)

        process = []
        for i in range(0, self.N-1):
            p = Process(target=self.combineSolution, args=(i,))
            p.start()
            process.append(p)

        for p in process:
            p.join()
        
        self.nu = np.average(self.nuBar, axis=0)

    def step_iterative(self):
        # Solve for X_t+1
        for i in range(0, self.N-1):
            t = self.solveIndividual(i)
            self.XBar[i] = t.T

        self.X = np.average(self.XBar, axis=0)
        # self.nu = np.average(self.nuBar, axis=0)

        self.X = self.X.reshape(-1, 1)
        self.nu = self.nu.reshape(-1, 1)

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - (self.alpha / self.rho) * np.sign(self.Z)

        # Combine
        for i in range(0, self.N-1):
            self.nuBar[i] = self.combineSolution(i)
        self.nu=np.average(self.nuBar,axis=0)

    def LassoObjective(self):
        return 0.5 * norm(self.A.dot(self.X) - self.b)**2 + self.alpha * norm(self.X, 1)
    
    def predict(self,test_X,test_y,classification=False):
        predict_y = np.matmul(test_X,self.X)
        if classification:
            predict_y=predict_y > 0.5
            print("Accuracy: ",accuracy_score(test_y,predict_y))
        else:
            print('Implemented R2 score: ',r2_score(test_y,predict_y))
            print('ScikitLearn MAE: ',mean_absolute_error(test_y,predict_y))
            print('ScikitLearn MSE: ',mean_squared_error(test_y,predict_y))
