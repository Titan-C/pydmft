# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:46:02 2015

@author: oscar

Translation of QMC Hirsch - Fye
"""
import numpy as np
Lrang=2**15
lfak=32

def g0(mu=0, beta=16., D=1):
    """Initiate green function"""
    fg0 = np.zeros(2*Lrang, dtype=np.complex)
    w = 1j*np.pi*np.arange(-Lrang+1, Lrang, 2) / beta
    sq = np.sqrt((w - mu)**2 - D)
    sig = np.sign(sq.imag*w.imag)
    fg0[1::2] = 2./(w-mu+sig*sq)
    return w, fg0


def g0t(g0):
    """Fourier transform into time"""
    g0t = np.fft.fft(g0)/16.
    g0t[::2] *= -1
    g0b = np.roll(g0t, Lrang)
    #trick to treat discontinuity
    g0b[Lrang] += 0.5
    g0b[0] = -g0b[Lrang]
    return g0t, g0b

def extract_g0t(g0t, lfak=32):
    """Extract a reducted amout of points of g0t"""

    dx = np.int(2.**15 / lfak)
    gt = np.concatenate( (g0t[Lrang::dx], [1.-g0t[Lrang]]))

    return np.concatenate((-gt[-1:0:-1], gt))

w, g=g0()
gt, gb=g0t(g)
g0 = extract_g0t(gb)
dtau, U = 0.5, 2
lamb=np.arccosh(np.exp(dtau*U/2))

def ising_v(lamb, polar=0.5):
    """initialize the vector v of Ising fields"""
    vis = np.ones(lfak)
    rand = np.random.rand(lfak)
    vis[rand>polar] = -1
    return vis*lamb

v=ising_v(lamb)