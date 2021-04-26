import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from aff_prop.aff_prop import affinity_propagation

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

# separate clusters
plt.scatter(*c1.T)
plt.scatter(*c2.T)
plt.scatter(*c3.T)
plt.show()

# preference and similarity
p = -10 * np.ones(n)
s = -distance_matrix(c, c) + np.diag(p)

k, exemplars, labels, evidence, i = affinity_propagation(s)
print(k)
print(exemplars)
print(labels)
print(evidence)
print(i)