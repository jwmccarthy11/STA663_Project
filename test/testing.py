import time
import numpy as np
from aff_prop import aff_prop
from scipy.sparse import coo_matrix

np.random.seed(1234)

n = 5000
p = 0.01

# random dense data with negative inf values
dense = np.random.rand(n, n) + np.random.choice([-np.inf, 0], size=(n, n), p=[1-p, p])

# dense matrix
s = dense
r = np.zeros_like(s)
a = np.zeros_like(s)

start = time.time()

r_dense = resp_dense(r, s, a)
a_dense = aval_dense(r_dense, a)

dense_time = time.time() - start

# row and col of non-negative-inf values
loc = np.argwhere(dense != -np.inf)

# coo input
row = loc[:, 0]
col = loc[:, 1]
val = dense[row, col]

# sparse matrix
s = coo_matrix((val, (row, col)), shape=(n, n)).tocsr()
r = s.copy()
a = s.copy()
r.data = 0. * s.data
a.data = 0. * s.data

start = time.time()

r_sparse = resp_sparse(r, s, a)
a_sparse = aval_sparse(r_sparse, a)

sparse_time = time.time() - start

print('\n---------------------------')
print(' Sparse Similarity Matrix')
print('---------------------------\n')
print('Full:\n')
print(dense)
print('\nCSR:\n')
print(s)

print('\n-------------------------')
print(' Updated responsibility')
print('-------------------------\n')
print('Full:\n')
print(r_dense)
print('\nCSR:\n')
print(r_sparse)

print('\n-------------------------')
print(' Updated availability')
print('-------------------------\n')
print('Full:\n')
print(a_dense)
print('\nCSR:\n')
print(a_sparse)

print('\nDense Single Iteration Runtime (s):\n', dense_time)
print('\nSparse Single Iteration Runtime (s):\n', sparse_time)