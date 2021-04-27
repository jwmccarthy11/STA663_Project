import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from aff_prop.aff_prop import affinity_propagation, plot_affinity_clusters, aval_dense, resp_dense
from sklearn.cluster import AffinityPropagation

np.random.seed(1234)

size = 100

# cluster 1
c1 = np.random.multivariate_normal(
    mean=np.array([0, -0.5]),
    cov=np.array([
        [0.1, 0],
        [0, 0.1]
    ]),
    size=size
)

# cluster 2
c2 = np.random.multivariate_normal(
    mean=np.array([1, -10]),
    cov=np.array([
        [0.1, 0],
        [0, 0.1]
    ]),
    size=size
)

# cluster 3
c3 = np.random.multivariate_normal(
    mean=np.array([2, 1]),
    cov=np.array([
        [0.1, 0],
        [0, 0.1]
    ]),
    size=size
)

# combined data
c = np.r_[c1, c2, c3]

# sample size
n = len(c)

p = np.ones(n) * -14.09090909
s = (-distance_matrix(c, c) + np.diag(p)).astype(np.float64)
r = np.zeros_like(s)
a = np.zeros_like(s)

aval_dense(r, a)

start = time.time()

k, ex, lbl, _, _ = affinity_propagation(s)

print(time.time() - start)

plot_affinity_clusters(c, ex, lbl, pca_axes=True)
plt.show()