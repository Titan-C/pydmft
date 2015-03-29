# -*- coding: utf-8 -*-
cdef extern from "cblas.h":
    enum CBLAS_ORDER: CblasRowMajor, CblasColMajor
    void lib_dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                                double *x, int dx, double *y, int dy,
                                double *A, int lda)
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


cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    double uniform "gsl_rng_uniform"(gsl_rng *r)

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef update(np.ndarray[np.float64_t, ndim=2] gup,
             np.ndarray[np.float64_t, ndim=2] gdw,
             np.ndarray[np.float64_t, ndim=1] v):
    cdef double dv, ratup, ratdw, rat
    cdef int j, i, up, dw, pair, N=v.shape[0]
    for j in range(N):
        dv = 2.*v[j]
        ratup = 1. + (1. - gup[j, j])*(exp(-dv)-1.)
        ratdw = 1. + (1. - gdw[j, j])*(exp( dv)-1.)
        rat = ratup * ratdw
        rat = rat/(1.+rat)
        if rat > uniform(r):
            v[j] *= -1.
            gnew(gup, v[j], j, 1.)
            gnew(gdw, v[j], j, -1.)
