import numpy as np
from numpy import log
from scipy.special import logsumexp
#import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

class BernoulliMixture:

    def __init__(self, n_clusters=4, n_iter=100, tolerance=1e-8, alpha1=1e-6, alpha2=1e-6):
        '''Basic Intiialization'''
        self._n_clusters = n_clusters
        self._n_iter = n_iter
        self._tolerance = tolerance
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        
        self.Pi = 1/self._n_clusters*np.ones(self._n_clusters)

    def _P(self, x, Mu, Pi):
        '''computes the log of the conditional probability of the latent variable given the data and Mu, Pi'''
        ll = np.dot(x,log(Mu))+np.dot(1-x,log(1-Mu))
        Z = (log(Pi)+ ll - logsumexp(ll+log(Pi), axis=1, keepdims=True))
        return Z

    def fit(self, data):
        global LogLikelihood
        '''carries out the EM iteration'''
        self._n_samples, self._n_features = data.shape
        self.Mu = np.random.uniform(.25, .75, size=self._n_clusters*self._n_features).reshape(self._n_features, self._n_clusters)
        N = self._n_samples

        for i in tqdm(range(self._n_iter)):
            eCP = np.exp(self._P(data, self.Mu, self.Pi)) # eCP = e^{Conditional_Probability}
            W = eCP/eCP.sum(axis=1,keepdims=True)
            R = np.dot(data.transpose(), W)
            Q = np.sum(W, axis=0, keepdims=True)
            LogLikelihood.append(Q)
            # print("Shape: ", W.shape, R.shape, Q)
            
            self._oldPi = self.Pi
            
            self.Mu = (R + self._alpha1)/(Q + self._n_features*self._alpha1)
            self.Pi = (Q + self._alpha2)/(N + self._n_clusters * self._alpha2)
            
            if np.allclose(self._oldPi, self.Pi):
                return

if __name__ == "__main__":
    LogLikelihood = []
    data = np.genfromtxt("A2Q1.csv", delimiter=',')
    model = BernoulliMixture(n_clusters=4)
    model.fit(data)
    avg = np.divide(np.sum(LogLikelihood, axis=0), 100)
    print(avg)
    plt.plot(avg.T)
    plt.xlabel("Likelihoods per K (averaged over 100 random initializations)")
    plt.ylabel("Log-Likelihood values")
    # plt.savefig("plots/plot-EM_Bernoulli.png")
    plt.show()

