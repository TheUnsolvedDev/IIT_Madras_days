import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm

class GMM:

    def __init__(self,X,K,iterations):
        self.iterations = iterations
        self.K = K
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None


    def run(self):
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        self.XY = np.array([x.flatten(),y.flatten()]).T
           
                    
        """ Set the initial mu, covariance and pi values"""
        self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.K,len(self.X[0]))) # (n x m) dim matrix
        self.cov = np.zeros((self.K,len(X[0]),len(X[0]))) # (n x m x m) dim covariance matrix for each source
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],5)

        self.pi = np.ones(self.K) / self.K

        log_likelihoods = []
                    
        for i in tqdm(range(self.iterations)):
            """E Step"""
            r_ik = np.zeros((len(self.X),len(self.cov)))

            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ik[0]))):
                co+=self.reg_cov
                mn = multivariate_normal(mean=m,cov=co)
                r_ik[:,r] = p*mn.pdf(self.X) / np.sum([pi_k*multivariate_normal(mean=mu_k,cov=cov_k).pdf(X) \
                                                     for pi_k,mu_k,cov_k in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)

            """M Step"""
            # Calculate new mean & new covariance_matrice
            self.mu = []
            self.cov = []
            self.pi = []
            
            for k in range(len(r_ik[0])):
                m_k = np.sum(r_ik[:,k],axis=0)
                mu_k = (1/m_k)*np.sum(self.X*r_ik[:,k].reshape(len(self.X),1),axis=0)
                self.mu.append(mu_k)

                # Calculate and append the covariance matrix per source based on the new mean
                cov_k = ((1/m_k)*np.dot((np.array(r_ik[:,k]).reshape(len(self.X),1)*(self.X-mu_k)).T,(self.X-mu_k)))+self.reg_cov
                self.cov.append(cov_k)
                # Calculate and append pi_new
                pi_k = m_k / np.sum(r_ik)
                self.pi.append(pi_k)


            """Log likelihood"""
            lh = np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(X) \
                                                  for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])
            llh = np.log(lh)
            log_likelihoods.append(llh)

            

        fig2 = plt.figure(figsize=(10,7))
        ax1 = fig2.add_subplot(111) 
        ax1.set_title('Log-Likelihood')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Loglikelihood")
        ax1.plot(range(0,self.iterations,1),log_likelihoods)
        # plt.savefig("plots/plot-EM_Gaussian.png")
        plt.show()

if __name__ == "__main__":
    data = np.genfromtxt("A2Q1.csv", delimiter=',')
    X = data
    print("Shape of X (dataset): ", np.shape(X))
    K = 4 # Given K=4
    model = GMM(X,K,100)     
    model.run()
