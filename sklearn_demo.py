import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from skcuda.linalg import PCA as cuPCA

# map the data to 4 dimensions
pca = cuPCA(n_components=4)

# 1000 samples of 100-dimensional data vectors
X = np.random.rand(1000,100)

# note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
X_gpu = gpuarray.GPUArray((1000,100), np.float64, order="F")

# copy data to gpu
X_gpu.set(X)

# calculate the principal components
T_gpu = pca.fit_transform(X_gpu)

# show that the resulting eigenvectors are orthogonal
dot_product = linalg.dot(T_gpu[:,0], T_gpu[:,1])
print(dot_product)
