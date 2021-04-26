import numpy as np
import pyximport
pyximport.install()


def resp_dense(r, s, a, lmb=0.5):
    """
    Compute updated dense responsibility matrix.
    """
    temp = c_resp_dense(s, np.add(a, s))
    return r * lmb + (1 - lmb) * temp


def resp_sparse(r, s, a, lmb=0.5):
    """
    Compute updated sparse responsibility matrix.
    """
    temp = a + s
    temp.data = c_resp_sparse(temp.indptr, temp.data, s.data)
    return r * lmb + (1 - lmb) * temp


def aval_dense(r, a, lmb=0.5):
    """
    Compute updated dense availability matrix
    """
    temp = c_aval_dense(r)
    return a * lmb + (1 - lmb) * temp


def aval_sparse(r, a, lmb=0.5):
    """
    Compute updated sparse availability matrix
    """
    temp = r.tocsc(copy=True)
    temp.data = c_aval_sparse(temp.indptr, temp.indices, temp.data)
    return a * lmb + (1 - lmb) * temp.tocsr()


def affinity_propagation(s, max_iter=100, max_conv_iter=10, lmb=0.5):
    """
    Performs affinity propagation clustering on a given similarity matrix.

    Parameters:
        s (np.ndarray or scipy.sparse) : similarity matrix
        max_iter (int) : max number of iterations
        max_conv_iter (int) : max number of iterations w/out change in output
        lmb (float) : update damping factor
    """
    if isinstance(s, np.ndarray):
        print(s)