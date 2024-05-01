import numpy as np
from utils import *

a = np.array([[1,2,3,4,5] for i in range(10)])

b = [np.dot(a[i],a[i].T) for i in range(10)]

print(a.T@a)
print(b)

a = np.array([[1,2,3,4,5]])
print(a.T@a)