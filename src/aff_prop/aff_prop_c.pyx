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
cpdef c_aval_dense(double[:, :] r):
    """
    Cython accelerated sparse availability update.
    """
    cdef int i, j, k
    cdef int n = r.shape[0]
    cdef double col_sum
    cdef double[:, :] aval = np.zeros_like(r)

    for i in range(n):
        col_sum = 0

        # find sum of column values
        for j in range(n):
            if i == j:
                col_sum += r[j, i]
            else:
                col_sum += max(0, r[j, i])

        # compute availability given sums
        for k in range(n):
            if i == k:
                aval[k, i] = (col_sum - r[k, i]) if r[k, i] > -INFINITY else -INFINITY
            else:
                aval[k, i] = min(0, col_sum - max(0, r[k, i]))

    return np.array(aval)