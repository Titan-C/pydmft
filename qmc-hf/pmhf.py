# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:46:02 2015

@author: oscar

Translation of QMC Hirsch - Fye
"""
import numpy as np
from scipy.linalg import solve
from scipy.interpolate import interp1d
Lrang = 2**15
lfak = 32


def matsubara_freq(beta=16., fer=1):
    return 1j*np.pi*np.arange(-Lrang+fer, Lrang, 2) / beta


def greenF(w, sigma=0, mu=0, beta=16., D=1):
    """Calculate green function lattice"""
    fg0 = np.zeros(2*Lrang, dtype=np.complex)
    zeta = w - mu - sigma
    sq = np.sqrt((zeta)**2 - D)
    sig = np.sign(sq.imag*w.imag)
    fg0[1::2] = 2./(zeta+sig*sq)
    return w, fg0


def FFT(gt,beta):
    """Fourier transfor into matsubara frequencies"""
    # trick to treat discontinuity
    gt[Lrang] -= 0.5
    gt[0] = -gt[Lrang]
    gt[::2] *= -1
    gw = np.fft.fft(gt)*beta/2/Lrang

    return gw


def iFFT(gw, beta=16.):
    """Inverse Fourier transform into time"""
    gt = np.fft.ifft(gw)*2*Lrang/beta
    gt[::2] *= -1
    # trick to treat discontinuity
    gt[Lrang] += 0.5
    gt[0] = -gt[Lrang]
    return gt.real


def dyson(g,g0):
    """Dyson equation for the self energy"""
    sigma = np.zeros(2*Lrang, dtype=np.complex)
    sigma[1::2] = 1/g0[1::2] - 1/g[1::2]
    return sigma


def extract_g0t(g0t, lfak=32):
    """Extract a reducted amout of points of g0t"""

    dx = np.int(2.**15 / lfak)
    gt = np.concatenate((g0t[Lrang::dx], [1.-g0t[Lrang]]))

    return np.concatenate((-gt[:-1], gt))


def ising_v(lamb, polar=0.5):
    """initialize the vector v of Ising fields"""
    vis = np.ones(lfak)
    rand = np.random.rand(lfak)
    vis[rand>polar] = -1
    return vis*lamb



def impurity(g0):
    g0[0] = -g0[lfak]

    gind = lfak + np.arange(lfak).reshape(-1, 1)-np.arange(lfak).reshape(1, -1)
    gx = g0[gind]

    gup = gnewclean(gx, 1.)
    gdw = gnewclean(gx, -1.)

    gstup, gstdw = mcs(1000, gup, gdw)

    return wrapup(gstup, gstdw)



def wrapup(gstup, gstdw):
    xgu = np.zeros(2*lfak+1)
    xgd = np.zeros(2*lfak+1)
    for i in range(1,lfak):
        xgu[i] = np.trace(gstup, offset=lfak-i)
        xgd[i] = np.trace(gstdw, offset=lfak-i)
    for i in range(lfak):
        xgu[i+lfak] = np.trace(gstup, offset=-i)
        xgd[i+lfak] = np.trace(gstdw, offset=-i)

    xga = (xgu + xgd) / 2.
    xg = np.zeros(2*lfak+1)
    xg[lfak+1:-1] = (xga[lfak+1:-1]-xga[1:lfak]) / lfak
    xg[1:lfak] = -xg[lfak+1:-1]
    xg[lfak] = xga[lfak] / lfak
    xg[0] = -xg[lfak]
    xg[-1] = 1-xg[lfak]

    return xg


def mcs(sweeps, gup, gdw):
    gstup, gstdw = np.zeros((lfak,lfak)), np.zeros((lfak,lfak))

    for mcs in xrange(sweeps):
        for j in xrange(lfak):
            dv = 2.*v[j]
            ratup = 1. + (1. - gup[j,j])*(np.exp(-dv)-1.)
            ratdw = 1. + (1. - gdw[j,j])*(np.exp( dv)-1.)
            rat = ratup * ratdw
            rat = rat/(1.+rat)
            if rat > np.random.rand():
                v[j] *= -1.
                gup = gnew(gup, j, 1.)
                gdw = gnew(gdw, j, -1.)

        gstup += gup
        gstdw += gdw

    gstup = gstup/sweeps
    gstdw = gstdw/sweeps

    return gstup, gstdw


def gnewclean(gx, sign):
    ee = np.exp(sign*v) - 1.
    b = np.eye(lfak) - ee * (gx-np.eye(lfak))

    return solve(b, gx)

def gnew(g, j, sign):
    dv = sign*v[j]*2
    ee = np.exp(dv)-1.
    a = ee/(1. + (1.-g[j, j])*ee)
    return g + a * (g[:, j] - np.eye(lfak)[:, j]).reshape(-1, 1) * g[j, :].reshape(1, -1)



def interpol(gt):
    t = np.linspace(0, 1, gt.size)
    f = interp1d(t, gt)
    tf = np.linspace(0, 1, Lrang+1)
    ngt = f(tf)
    ngt = np.concatenate((-ngt[:-1], ngt))

    return ngt[:-1]

dtau, U = 0.5, 2.5
lamb = np.arccosh(np.exp(dtau*U/2))
w = matsubara_freq()
Gw = greenF(w)
G0t = iFFT(Gw)
g0t = extract_g0t(G0t)

v=ising_v(lamb)
gx = impurity(g0t)
neg=interpol(gx[lfak:])