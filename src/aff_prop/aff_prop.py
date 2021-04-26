import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from . import aff_prop_c


def resp_dense(r, s, a, lmb=0.5):
    """
    Compute updated dense responsibility matrix.
    """
    temp = aff_prop_c.c_resp_dense(np.add(a, s), s)
    return r * lmb + (1 - lmb) * temp


def resp_sparse(r, s, a, lmb=0.5):
    """
    Compute updated sparse responsibility matrix.
    """
    temp = a + s
    temp.data = aff_prop_c.c_resp_sparse(temp.indptr, temp.data, s.data)
    return r * lmb + (1 - lmb) * temp


def aval_dense(r, a, lmb=0.5):
    """
    Compute updated dense availability matrix
    """
    temp = aff_prop_c.c_aval_dense(r)
    return a * lmb + (1 - lmb) * temp


def aval_sparse(r, a, lmb=0.5):
    """
    Compute updated sparse availability matrix
    """
    temp = r.tocsc(copy=True)
    temp.data = aff_prop_c.c_aval_sparse(temp.indptr, temp.indices, temp.data)
    return a * lmb + (1 - lmb) * temp.tocsr()


def affinity_propagation(s, max_iter=100, max_conv_iter=10, tol=1e-6, lmb=0.5):
    """
    Performs affinity propagation clustering on a given similarity matrix.

    Parameters:
        s (np.ndarray) : similarity matrix
        max_iter (int) : max number of iterations
        max_conv_iter (int) : max number of iterations w/out change in output
        tol (float) : stopping condition tolerance
        lmb (float) : update damping factor
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
        new_message = resp + aval

        # tolerance stopping condition
        if np.all(np.abs(old_message - new_message) < tol):
            break

        # find exemplars and stop if the same max_conv_iter times
        exem_new = new_message.argmax(axis=1)
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

    return k, exemplars, labels, new_message, i