"""KPCA
Author: Jashaswimalya Acharjee
Roll: CS22E005
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

data = np.genfromtxt('./Dataset.csv', delimiter=',')

class kernel:

    def __init__(self, sigma = 1, d = 2):
        self.sigma = sigma
        self.d = d

    def polynomial(self, x, y):
        """
        Equation: k(x, y) = (1 + x.T * y)^d
        """
        return (1 + (x.T@y))**self.d

    def exp(self, x, y):
        """
        Equation: k(x, y) = exp(- ||x-y|| / (2 * sigma^2))
        """
        return np.exp(- (1/ (2*self.sigma**2)) * (np.linalg.norm(x-y))**2)

class KPCA:
    def __init__(self, X, kernel, d):
        """
        Kernel PCA Class

        Params:
            X: d x n Matrix
            kernel: Kernel Function
            d: Number of principal components to be chosen
        """
        self.X = X
        self.kernel = kernel 
        self.d = d
    
    def __kernel_matrix(self):
        """
        Computes Kernel Matrix

        Output:
            K: nxn matrix
        """
        K = []
        _, c = self.X.shape
        for fil in range(c):
            k_aux = []
            for col in range(c):
                k_aux.append(self.kernel(self.X[:, fil], self.X[:, col]))
            K.append(k_aux)
        K = np.array(K)
        # Centering K
        ones = np.ones(K.shape)/c
        K = K - ones@K - K@ones + ones@K@ones
        return K
    
    def __eig_decompK(self):
        """
        Decomposition of K

        Output: 
            (eigen_value, eigenvector): List of tuples ordered by Eigen_Values in Descending order. 
        """
        self.K = self.__kernel_matrix()
        eigval, eigvec = np.linalg.eig(self.K)

        # Normalize eigenvectors and compute singular values of K
        eig_val_vec = [(np.sqrt(eigval[i]), eigvec[:,i]/np.sqrt(eigval[i]) ) for i in range(len(eigval))]
        eig_val_vec.sort(key=lambda x: x[0], reverse=True)
        return eig_val_vec
    
    def project(self):
        """
        Compute scores

        Output:
            scores: T = sigma * (V_d).T
        """
        self.eig_val_vec = self.__eig_decompK()
        eig_val_vec_dim = self.eig_val_vec[:self.d]
        self.sigma = np.diag([i[0] for i in eig_val_vec_dim])
        self.v = np.array([list(j[1]) for j in eig_val_vec_dim]).T
        self.sigma = np.real_if_close(self.sigma, tol=1)
        self.v = np.real_if_close(self.v, tol=1)
        self.scores = self.sigma @ self.v.T
        return self.scores

    def plot_scores(self, grid = True):
        # Dummy Proofing Check
        if self.d < 2:
            print("Not enough principal components")
            return

        plt.scatter(self.scores[0,:], self.scores[1,:], c = 'blue')
        plt.grid(grid)
        plt.title('KPCA')
        plt.xlabel('$1^{st}$ Principal Component')
        plt.ylabel('$2^{nd}$ Principal Component')
        #plt.savefig("plots/plot-"+ datetime.datetime.now().strftime("%f") + ".png")
        plt.show()

"""# Polynomial"""

# For d -> {2, 3}
for i in range(2, 4):
  # print("For d = " , i)
  X = data.T
  K = kernel(d = i).polynomial
  k = 2
  kpca = KPCA(X, K, k)
  scores = kpca.project()
  kpca.plot_scores(grid = True)

"""# Exponential (Gaussian)"""

# For sigma -> {0.1, 0.2, ..., 1}
for i in range(1, 11):
  i = i / 10
  # print("For Sigma = ",i)
  X = data.T
  K = kernel(sigma = i).exp # Kernel
  k = 2 # Number of PCA Components
  kpca = KPCA(X, K, k)
  scores = kpca.project()
  kpca.plot_scores(grid = True)

T = kpca.scores # Matrix of scores
K = kpca.K # Kernel matrix
V = kpca.v # Matrix of eigenvectors
S = kpca.sigma # Diagonal matrix of (real) eigen values

print("Shape of Matrix of Scores", T.shape)
print("Shape of Kernel Matrix", K.shape)
print("Shape of Matrix of Eigen Vector", V.shape)
print("Shape of Diagonal Matrix", S.shape)
