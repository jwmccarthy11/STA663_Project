import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from . import aff_prop_c


def resp_dense(r, s, a, lmb=0.5):
    """
    Compute updated dense responsibility matrix.
    """
    temp = aff_prop_c.c_resp_dense(np.add(a, s), s)
    return r * lmb + (1 - lmb) * temp


def aval_dense(r, a, lmb=0.5):
    """
    Compute updated dense availability matrix
    """
    temp = aff_prop_c.c_aval_dense(r)
    return a * lmb + (1 - lmb) * temp


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
    n = s.shape[0]
    stop = max_conv_iter

    # initialize relevant matrices
    if not isinstance(s, np.ndarray):
        raise TypeError('Similarity matrix s is not numpy/scipy.sparse matrix')
    resp = np.zeros_like(s)
    aval = np.zeros_like(s)
    exem = np.zeros(n, dtype=int)

    for i in range(max_iter):
        old_message = resp + aval
        resp = resp_dense(resp, s, aval, lmb)
        aval = aval_dense(resp, aval, lmb)
        evidence = resp + aval

        # tolerance stopping condition
        if np.all(np.abs(old_message - evidence) < tol):
            break

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
    if pca_axes:
        X_new = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X = pca.fit_transform(X_new)
    plt.scatter(X[:, param_x], X[:, param_y], c=labels, cmap="viridis")
    plt.scatter(X[exemplars, param_x], X[exemplars, param_y], c="red")