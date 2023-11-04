from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

def stoch_grad(A,X,B, batch_size=100):
    batch_no=0
    der_list=[]
    for i in range(0,len(A),batch_size):
        batch_no+=1
        error=A[i:i+100,:]*X-B[i:i+100,:]
        err_sum=np.sum(error)
        derivative=[]
        for k in range(len(X)):
            derivative.append(2*X[k,0]*err_sum/batch_size)
        derivative=np.matrix(derivative).reshape(100,1)
        der_list.append(derivative)
    final=sum(der_list) / batch_no
    return final
    

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
    alpha=0.01
    
    
    for i in tqdm(range(epoch)):
        s_grad=stoch_grad(A,X,B)
        X=X-(alpha*s_grad)
    #print(X)
    
    w_ml=X
    
    X=np.random.rand(100,1)
    l2_norm=[]
    for i in tqdm(range(epoch)):
        s_grad=stoch_grad(A,X,B)
        X=X-(alpha*s_grad)
        tmp=np.linalg.norm(X-w_ml,ord=2)
        l2_norm.append(tmp)
    
    print(l2_norm)
    
    fig = plt.figure(figsize=(10,10))
    timestep=[i+1 for i in range(epoch)]
    plt.plot(timestep,l2_norm)
    plt.show()
