import time
import numpy as np
from aff_prop import resp_sparse, resp_dense
from scipy.sparse import csr_matrix, coo_matrix


np.random.seed(1234)

n = 5
p = 0.5

# random dense data with negative inf values
dense = np.random.rand(n, n) + np.random.choice([-np.inf, 0], size=(n, n), p=[1-p, p])

# dense matrix
s = dense
r = np.zeros_like(s)
a = np.zeros_like(s)

start = time.time()

dense_out = resp_dense(r, s, a)

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

sparse_out = resp_sparse(r, s, a)

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
print(dense_out)
print('\nCSR:\n')
print(sparse_out)

print(dense_time)
print(sparse_time)