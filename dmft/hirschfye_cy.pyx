# -*- coding: utf-8 -*-
cdef extern from "cblas.h":
    enum CBLAS_ORDER: CblasRowMajor, CblasColMajor
    void lib_dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                                double *x, int dx, double *y, int dy,
                                double *A, int lda)
import numpy as np
cimport numpy as np
import cython
from libc.math cimport exp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef gnew(np.ndarray[np.float64_t, ndim=2] g, double v, int k, double sign):
    cdef double dv, ee, alpha
    cdef int N = g.shape[0]
    dv = sign*v*2
    ee = exp(dv)-1.
    alpha = ee/(1. + (1.-g[k, k])*ee)
    cdef np.ndarray[np.float64_t, ndim=1] x = g[:, k].copy()
    cdef np.ndarray[np.float64_t, ndim=1] y = g[k, :].copy()

    x[k] -= 1.
    lib_dger(CblasColMajor, N, N, alpha,
            &x[0], 1, &y[0], 1, &g[0,0], N)
