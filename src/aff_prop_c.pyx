import cython
import numpy as np
cimport numpy as cnp


cdef extern from "math.h":
    double INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_resp_dense(double[:, :] data, double[:, :] s):
    """
    Cython accelerated dense responsibility update.
    """
    cdef int i, j, k
    cdef int idx_s, idx_e, idx_m
    cdef int n = data.shape[0]
    cdef double max_1, max_2
    cdef double[:, :] resp = np.zeros_like(data)

    for i in range(n):
        # reset max values
        max_1 = max_2 = -INFINITY

        # find two highest values in row
        for j in range(n):
            if data[i, j] >= max_1:
                max_2 = max_1
                max_1 = data[i, j]
                idx_m = j
            elif max_2 <= data[i, j] < max_1:
                max_2 = data[i, j]

        # assign maximum to output
        for k in range(n):
            if k != idx_m:
                # account for rows with only -inf values
                resp[i, k] = s[i, k] - max_1 if max_1 > -INFINITY else -INFINITY
            else:
                # account for rows with only one val != -inf
                resp[i, k] = s[i, k] - max_2 if max_2 > -INFINITY else -INFINITY

    return np.array(resp)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_resp_sparse(int[:] indptr, double[:] data, double[:] s):
    """
    Cython accelerated sparse responsibility update.
    """
    cdef int i, j, k
    cdef int idx_s, idx_e, idx_m
    cdef int npr = indptr.shape[0]
    cdef double max_1, max_2
    cdef double[:] resp = np.zeros_like(data)

    for i in range(npr-1):
        idx_s = indptr[i]
        idx_e = indptr[i+1]

        # reset max values
        max_1 = max_2 = -INFINITY

        # find two highest values in row
        for j in range(idx_s, idx_e):
            if data[j] >= max_1:
                max_2 = max_1
                max_1 = data[j]
                idx_m = j
            elif max_2 <= data[j] < max_1:
                max_2 = data[j]

        # assign maximum to output
        for k in range(idx_s, idx_e):
            if k != idx_m:
                resp[k] = s[k] - max_1
            else:
                # account for rows with only one val != -inf
                resp[k] = s[k] - max_2 if max_2 > -INFINITY else -INFINITY

    return np.array(resp)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_aval_sparse(int[:] indptr, int[:] rowidx, double[:] data):
    """
    Cython accelerated sparse availability update.
    """
    cdef int i, j, k
    cdef int idx_s, idx_e
    cdef int npr = indptr.shape[0]
    cdef double[:] sums = np.zeros(npr-1)
    cdef double[:] aval = np.zeros_like(data)

    # loop over indptr pairs
    for i in range(npr-1):
        # find sum of column max values
            # don't max diagonal elements w/ 0
        # loop over column
            # subtract cell value if not diagonal
            # min cell value w/ 0

        pass