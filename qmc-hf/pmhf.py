# -*- coding: utf-8 -*-
"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm
"""
import numpy as np
from scipy.linalg import solve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import jit, autojit
from dmft.common import matsubara_freq, fft, ifft


def greenF(w, sigma=0, mu=0, D=1):
    """Calculate green function lattice"""
    Gw = np.zeros(2*w.size, dtype=np.complex)
    zeta = w - mu - sigma
    sq = np.sqrt((zeta)**2 - D)
    sig = np.sign(sq.imag*w.imag)
    Gw[1::2] = 2./(zeta+sig*sq)
    return Gw


def dyson_sigma(g, g0, fer=1):
    """Dyson equation for the self energy"""
    sigma = np.zeros(g.size, dtype=np.complex)
    sigma[fer::2] = 1/g0[fer::2] - 1/g[fer::2]
    return sigma


def dyson_g0(g, sigma, fer=1):
    """Dyson equation for the bare Green function"""
    g0 = np.zeros(g.size, dtype=np.complex)
    g0[fer::2] = 1/(1/g[fer::2] + sigma[fer::2])
    return g0


def extract_g0t(g0t, lfak=32):
    """Extract a reducted amout of points of g0t"""
    Lrang = g0t.size // 2
    dx = np.int(Lrang / lfak)
    gt = np.concatenate((g0t[Lrang::dx], [1.-g0t[Lrang]]))

    return np.concatenate((-gt[:-1], gt))


def ising_v(lamb, L=32, polar=0.5):
    """initialize the vector V of Ising fields

    .. math:: V = \\lambda (\\sigma_1, \\sigma_2, \\cdots, \\sigma_L)

    where the vector entries :math:`\\sigma_n=\\pm 1` are randomized subject
    to a threshold given by polar

    Parameters
    ----------
    lamb : float
        :math:`\\lambda` factor giving scaling to vector
    L : integer
        length of the array
    polar : float :math:`\\in (0, 1)`
        polarization threshold, probability of :math:`\\sigma_n=+ 1`

    Returns
    -------
    out : single dimension ndarray
    """
    vis = np.ones(L)
    rand = np.random.rand(L)
    vis[rand>polar] = -1
    return vis*lamb



def hf_solver(g0, v, sweeps):
    """Impurity solver call.
    Takes the propagator :math:`\\mathcal{G}^0(\\tau)` and transforms it
    into a discretized matrix as

    .. math:: \\mathcal{G}^0_{ij} = \\mathcal{G}^0(i\\Delta\\tau - j\\Delta\\tau)

    """
    lfak = v.size
    g0[0] = -g0[lfak]

    gind = lfak + np.arange(lfak).reshape(-1, 1)-np.arange(lfak).reshape(1, -1)
    gx = g0[gind]

    gup = gnewclean(gx, v, 1.)
    gdw = gnewclean(gx, v, -1.)

    gstup, gstdw = mcs(sweeps, gup, gdw, v)

    return wrapup(gstup, gstdw)


def wrapup(gstup, gstdw):
    lfak = gstdw.shape[0]
    xgu = np.zeros(2*lfak+1)
    xgd = np.zeros(2*lfak+1)
    for i in range(1, lfak):
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


def mcs(sweeps, gup, gdw, v):
    lfak = v.size
    gstup, gstdw = np.zeros((lfak, lfak)), np.zeros((lfak, lfak))

    for mcs in xrange(sweeps):
        for j in xrange(lfak):
            dv = 2.*v[j]
            ratup = 1. + (1. - gup[j, j])*(np.exp(-dv)-1.)
            ratdw = 1. + (1. - gdw[j, j])*(np.exp( dv)-1.)
            rat = ratup * ratdw
            rat = rat/(1.+rat)
            if rat > np.random.rand():
                v[j] *= -1.
                gup = gnew(gup, v, j, 1.)
                gdw = gnew(gdw, v, j, -1.)

        gstup += gup
        gstdw += gdw

    gstup = gstup/sweeps
    gstdw = gstdw/sweeps

    return gstup, gstdw


def gnewclean(gx, v, sign):
    """Returns the interacting function

    .. math:: G'_{ij} = B^{-1}_{ij}G_{ij}
    .. math:: u_j = \\exp(\\sigma v_j) - 1
    .. math:: B_{ij} = \\delta_{ij} - u_j ( G_{ij} - \\delta_{ij}) \\text{no sumation on } j

    """
    ee = np.exp(sign*v) - 1.
    ide = np.eye(v.size)
    b = ide - ee * (gx-ide)

    return solve(b, gx)


def gnew(g, v, j, sign):
    dv = sign*v[j]*2
    ee = np.exp(dv)-1.
    a = ee/(1. + (1.-g[j, j])*ee)
    return g + a * (g[:, j] - np.eye(lfak)[:, j]).reshape(-1, 1) * g[j, :].reshape(1, -1)



def interpol(gt, Lrang):
    t = np.linspace(0, 1, gt.size)
    f = interp1d(t, gt)
    tf = np.linspace(0, 1, Lrang+1)
    ngt = f(tf)
    ngt = np.concatenate((-ngt[:-1], ngt))

    return ngt[:-1]

dtau, U = 0.5, 2.5
lamb = np.arccosh(np.exp(dtau*U/2))
w = matsubara_freq()
G0w = greenF(w)


v = ising_v(lamb)

for i in range(4):
    G0t = ifft(G0w)
    g0t = extract_g0t(G0t)
    gt = hf_solver(g0t, v, 2000)
    plt.plot(gt, label='it {}'.format(i))

    Gt = interpol(gt[v.size:], 2**15)
#    G0t = interpol(g0t[lfak:])

    Gw = fft(Gt)
#    G0w = FFT(G0t)
#    sigma = dyson_sigma(Gw, G0w)
#
#    Gw = greenF(w, sigma[1::2])
#    G0w = dyson_g0(Gw, sigma)
    G0w = np.zeros(Gw.size, dtype=np.complex)
    G0w[1::2] = 1/(w - .25*Gw[1::2])


plt.legend(loc=0)
