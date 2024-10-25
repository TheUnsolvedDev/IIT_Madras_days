import numpy as np
from matplotlib import pyplot as plt

def find_grad(A,X,B):
    grad=[]
    der_sum=np.sum(A @ X-B)
    for i in range(len(X)):
        grad.append(2 * X[i,0]*der_sum/10000)
    grad=np.matrix(grad).reshape(100,1)
    return grad

if __name__ == "__main__":
    data_train = np.genfromtxt("A2Q2Data_train.csv", delimiter=',')
    print(data_train.shape)
    
    X = np.random.rand(100,1)
    A = data_train[:,0:-1]
    B = data_train[:, 100].reshape(len(data_train),1)
    print(X.shape)
    print(A.shape)
    print(B.shape)
    
    epoch=1000
    alpha=0.001
    
    for i in range(epoch):
        grad=find_grad(A,X,B)
        X=X-(alpha*grad)
    #print(X)
    
    w_ml=X
    
    X=np.random.rand(100,1)
    l2_norm=[]
    for i in range(epoch):
        grad=find_grad(A,X,B)
        X=X-(alpha*grad)
        values=np.linalg.norm(X-w_ml,ord=2)
        l2_norm.append(values)
    
    fig = plt.figure(figsize=(4,4))
    timestep=[i+1 for i in range(epoch)]
    ax = fig.add_subplot(111)
    plt.plot(timestep,l2_norm)
    plt.show()

# It is  observed that the difference gradually decreases with each iteration of algorithm.
