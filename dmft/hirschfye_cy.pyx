# -*- coding: utf-8 -*-

cimport numpy as np
import cython
from libc.math cimport exp, sqrt
from libcpp cimport bool

cdef extern from "hfc.h":
    void cgnew(size_t N, double *g, double dv, int k)

def gnew(np.ndarray[np.float64_t, ndim=2] g, double dv, int k):
    cdef int N=g.shape[0]
    cgnew(N, &g[0,0], dv, k)


cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    double uniform "gsl_rng_uniform"(gsl_rng *r)
    void cyset_seed "gsl_rng_set"(gsl_rng *r, unsigned long int)

cdef extern from "gsl/gsl_randist.h":
    double normal "gsl_ran_gaussian"(gsl_rng *r, double sigma)

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

def set_seed(seed=84263):
    cyset_seed(r, seed)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def updateDHS(np.ndarray[np.float64_t, ndim=2] gup,
              np.ndarray[np.float64_t, ndim=2] gdw,
              np.ndarray[np.float64_t, ndim=1] v,
              bool Heatbath = True):
    cdef double dv, ratup, ratdw, rat
    cdef int j, i, up, dw, pair, N=v.shape[0], acc = 0, nrat = 0
    for j in range(N):
        dv = -2.*v[j]
        ratup = 1. + (1. - gup[j, j])*(exp( dv)-1.)
        ratdw = 1. + (1. - gdw[j, j])*(exp(-dv)-1.)
        rat = ratup * ratdw
        if rat<0:
            nrat += 1
        if Heatbath:
            rat = rat/(1.+rat)

        if rat > uniform(r):
            acc += 1
            v[j] *= -1.
            cgnew(N, &gup[0,0],  dv, j)
            cgnew(N, &gdw[0,0], -dv, j)
    return acc, nrat
