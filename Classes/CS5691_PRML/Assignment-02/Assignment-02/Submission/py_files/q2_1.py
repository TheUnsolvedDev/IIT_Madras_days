import numpy as np

data_train = np.genfromtxt("A2Q2Data_train.csv", delimiter=',')
print(data_train.shape)

#ANALYTICAL SOLUTION USING GEOMETRC VIEW OR LINEAR ALGEBRA:
X=data_train[:,0:-1]
Y=data_train[:, 100].reshape(len(data_train),1)

print(X.shape)
print(Y.shape)

# w_ml
w_ml=np.linalg.pinv(X.T @ X) @ (X.T @ Y) # w_ml = (A^T A)^{-1} * A^T B
w_ml.shape

# Testing on the test data and reporting the SSE:
data_test = np.genfromtxt("A2Q2Data_test.csv", delimiter=',')
print(data_test.shape)

x_test=data_test[:,0:-1]
y_test=data_test[:, 100].reshape(len(data_test),1)
print(x_test.shape)
print(y_test.shape)

error=np.linalg.norm((x_test @ w_ml) - y_test)
print("SSE Error: ",error)
