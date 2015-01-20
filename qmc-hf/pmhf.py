# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:46:02 2015

@author: oscar

Translation of QMC Hirsch - Fye
"""
import numpy as np
from scipy.linalg import solve
Lrang = 2**15
lfak = 32


def g0(mu=0, beta=16., D=1):
    """Initiate green function"""
    fg0 = np.zeros(2*Lrang, dtype=np.complex)
    w = 1j*np.pi*np.arange(-Lrang+1, Lrang, 2) / beta
    sq = np.sqrt((w - mu)**2 - D)
    sig = np.sign(sq.imag*w.imag)
    fg0[1::2] = 2./(w-mu+sig*sq)
    return w, fg0


def g0t(g0, beta=16.):
    """Fourier transform into time"""
    g0t = np.fft.fft(g0)/beta
    g0t[::2] *= -1
    g0b = np.roll(g0t.real, Lrang)
    # trick to treat discontinuity
    g0b[Lrang] += 0.5
    g0b[0] = -g0b[Lrang]
    return g0b.real


def extract_g0t(g0t, lfak=32):
    """Extract a reducted amout of points of g0t"""

    dx = np.int(2.**15 / lfak)
    gt = np.concatenate((g0t[Lrang::dx], [1.-g0t[Lrang]]))

    return np.concatenate((-gt[-1:0:-1], gt))

w, g = g0()
gb = g0t(g)
g0 = extract_g0t(gb)
dtau, U = 0.5, 2.5
lamb = np.arccosh(np.exp(dtau*U/2))


def ising_v(lamb, polar=0.5):
    """initialize the vector v of Ising fields"""
    vis = np.ones(lfak)
    rand = np.random.rand(lfak)
    vis[rand>polar] = -1
    return vis*lamb

v=ising_v(lamb)

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

gx = impurity(g0)
from scipy.interpolate import interp1d
def interpol(gt):
    t = np.linspace(0, 1, gt.size)
    f = interp1d(x, gt)
    tf = np.linspace(0, 1, Lrang+1)
    ngt = f(tf)
    return np.concatenate((-ngt[-1:0:-1], ngt))


neg=interpol(gx[lfak:])