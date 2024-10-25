"""
# EigenFaces (PCA) on LFW Dataset
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_gallery(images, h, w, n_row = 5, n_col = 5): 
    plt.figure(figsize =(1.8 * n_col, 2.4 * n_row)) 
    plt.subplots_adjust(bottom = 0, left =.01, right =.99, top =.90, hspace =.35) 
    for i in range(n_row * n_col): 
        plt.subplot(n_row, n_col, i + 1) 
        plt.imshow(images[i].reshape((h, w)), cmap = plt.cm.gray) 
        plt.xticks(()) 
        plt.yticks(())

lfw_dataset = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
_, h, w = lfw_dataset.images.shape
X = lfw_dataset.data
print(X.shape)
plot_gallery(X,h,w)

n_components = 25
pca = PCA(n_components=n_components, whiten=True).fit(X)
plot_gallery(pca.components_,h,w)

"""
# EigenFaces on My Face
"""

import cv2

im = cv2.imread('test2.jpg',0)

plt.imshow(im,cmap='gray')
print(im.shape)
print(im.flatten().shape)

im_transform = pca.transform(im.flatten().reshape((1,1850)))
print(im_transform.shape)
im_recons = np.matmul(pca.components_.T,test.T)
im_recons = im_recons.reshape(50,37)
print(im_recons.shape)
plt.imshow(im_recons, cmap='gray')
plt.show()
