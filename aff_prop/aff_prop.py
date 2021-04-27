import numpy as np
import matplotlib.pyplot as plt
import aff_prop.aff_prop_c as apc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def resp_dense(r, s, a, lmb=0.5):
    """
    Compute updated dense responsibility matrix.
    """
    apc.c_resp_dense(r.shape[0], np.add(a, s), s, r, lmb)
    return r


def aval_dense(r, a, lmb=0.5):
    """
    Compute updated dense availability matrix
    """
    apc.c_aval_dense(r.shape[0], r, a, lmb)
    return a


def affinity_propagation(s, max_iter=100, max_conv_iter=10, tol=1e-6, lmb=0.5):
    """
    Performs affinity propagation clustering on a given similarity matrix.

    Parameters:
        s (np.ndarray) : similarity matrix
        max_iter (int) : max number of iterations
        max_conv_iter (int) : max number of iterations w/out change in output
        tol (float) : stopping condition tolerance
        lmb (float) : update damping factor

    Returns:
        k (int) : number of exemplars found
        exemplars (np.ndarray) : indices of exemplars
        labels (np.ndarray) : label corresponding to each points exemplar
        evidence (np.ndarray) : total evidence for exemplars
        i (int) : number of iterations run
    """
    if s.shape[0] != s.shape[1]:
        raise ValueError('Similarity matrix s is not square')

    n = s.shape[0]
    stop = max_conv_iter

    # initialize relevant matrices
    if not isinstance(s, np.ndarray):
        raise TypeError('Similarity matrix s is not numpy/scipy.sparse matrix')

    resp = np.zeros_like(s)
    aval = np.zeros_like(s)
    exem = np.zeros(n, dtype=int)

    for i in range(max_iter):
        resp = resp_dense(resp, s, aval, lmb)
        aval = aval_dense(resp, aval, lmb)
        evidence = resp + aval

        # find exemplars and stop if the same max_conv_iter times
        exem_new = evidence.argmax(axis=1)
        if np.all(exem_new == exem):
            stop -= 1
            if not stop:
                break
        else:
            stop = max_conv_iter
        exem = exem_new

    # find exemplars for each data point
    exemplars, labels = np.unique(exem, return_inverse=True)
    k = len(exemplars)

    return k, exemplars, labels, evidence, i


def plot_affinity_clusters(X, exemplars, labels, param_x=0, param_y=1, pca_axes=False):
    """
    Plots output of affinity propagation in 2D

    Parameters:
        X (np.ndarray) : the data that was clustered
        exemplars (np.ndarray) : points to be used as exemplars
        labels (np.ndarray) : label corresponding to each point's exemplar
        param_x (int) : the column index of the paramater used as the X value
        param_y (int) : the column index of the paramater used as the Y value
        pca_axes (bool) : option to use first and second principal components as axes
    """
    # create pca axes
    if pca_axes:
        X_new = StandardScaler().fit_transform(X)
        X = PCA(n_components=2).fit_transform(X_new)

    # create plot
    plt.scatter(X[:, param_x], X[:, param_y], c=labels, cmap="viridis")
    plt.scatter(X[exemplars, param_x], X[exemplars, param_y], c="red")