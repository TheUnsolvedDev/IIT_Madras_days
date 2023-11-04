"""PCA
Author: Jashaswimalya Acharjee
Roll: CS22E005
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

"""# Loading Data """

data = np.genfromtxt('./Dataset.csv', delimiter=',')

x = data[:,0]
y = data[:,1]

plt.scatter(x,y)

"""# Normalization """

### Calculate Mean
x_mean = np.mean(x)
y_mean = np.mean(y)
x_std = np.std(x)
y_std = np.std(y)

### Mean Normalize
x = (x - x_mean) / x_std
y = (y - y_mean) / y_std

"""# Covariance Matrix"""
cov_mat = np.cov([x,y])
print(cov_mat)

"""# EigenValue Decomposition """

# Eigenvectors and Eigenvalues for the from the covariance matrix
eig_val_c, eig_vec_c = np.linalg.eig(cov_mat)

# Make a paired list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_c[i]), eig_vec_c[:,i]) for i in range(len(eig_val_c))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

matrix_w = np.hstack((eig_pairs[0][1].reshape(2,1), eig_pairs[1][1].reshape(2,1)))
features = np.vstack([x,y])
transformed = np.dot(matrix_w,features)

"""# Usual Stats to monitor"""

#print(40 * '-')

# # Confirm that the EigenValue Pairs are correctly sorted by decreasing eigenvalues
#for i in eig_pairs:
#    print(i[1])

#print(40 * '-')
#print("Shape x,y")
#print(x.shape)
#print(y.shape)

#print(40 * '-')
#print('Matrix W: \n', matrix_w)
#print(matrix_w.shape)

#print(40 * '-')
#print('Features : \n', features)
#print(features.shape)

#print(40 * '-')
#print('Transformed : \n', transformed)
#print(transformed.shape)

plt.scatter(transformed[0,:],transformed[1,:])
plt.title('PCA')
plt.xlabel('$1^{st}$ Principal Component')
plt.ylabel('$2^{nd}$ Principal Component')
#plt.savefig("plots/plot-"+ datetime.datetime.now().strftime("%f") + ".png")
plt.show()
