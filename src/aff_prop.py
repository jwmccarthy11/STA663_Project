import numpy as np
from aff_prop_c import c_resp_dense, c_resp_sparse


def resp_dense(r, s, a, lmb=0.5):
    """
    Compute updated dense responsibility matrix.
    """
    new_r_mat = c_resp_dense(s, np.add(a, s))
    return r * lmb + (1 - lmb) * new_r_mat


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
    pass


def aval_sparse(r, a, lmb=0.5):
    """
    Compute updated sparse availability matrix
    """
    r = r.tocsc()
    a = a.tocsc()
    temp = c_aval_sparse(r.indptr, r.data, a.data)
    return a * lmb + (1 - lmb) * temp


def affinity_propagation(sparse=False):
    pass