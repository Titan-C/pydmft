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
Lrang = 2**15
lfak = 32


def matsubara_freq(beta=16., fer=1, Lrang=2**15):
    """Calculates an array containing the matsubara frequencies under the
    formula

    .. math:: i\\omega_n = i\\frac{\\pi(2n + f)}{\\beta}

    where :math:`i` is the imaginary unit and :math:`f=1` in the case of
    fermions, and zero for bosons

    Parameters
    ----------
    beta : float
            Inverse temperature of the system
    fer : 0 or 1 integer
            dealing with fermionic particles
    Lrang : integer
            size of the array : amount of matsubara frequencies

    Returns
    -------
    out : complex ndarray

    """
    return 1j*np.pi*np.arange(-Lrang+fer, Lrang, 2) / beta


def greenF(w, sigma=0, mu=0, D=1):
    """Calculate green function lattice"""
    Gw = np.zeros(2*Lrang, dtype=np.complex)
    zeta = w - mu - sigma
    sq = np.sqrt((zeta)**2 - D)
    sig = np.sign(sq.imag*w.imag)
    Gw[1::2] = 2./(zeta+sig*sq)
    return Gw


def FFT(gt, beta=16.):
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


def dyson_sigma(g, g0, fer=1):
    """Dyson equation for the self energy"""
    sigma = np.zeros(2*Lrang, dtype=np.complex)
    sigma[fer::2] = 1/g0[fer::2] - 1/g[fer::2]
    return sigma


def dyson_g0(g, sigma, fer=1):
    """Dyson equation for the bare Green function"""
    g0 = np.zeros(2*Lrang, dtype=np.complex)
    g0[fer::2] = 1/(1/g[fer::2] + sigma[fer::2])
    return g0


def extract_g0t(g0t, lfak=32):
    """Extract a reducted amout of points of g0t"""

    dx = np.int(2.**15 / lfak)
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



def hf_solver(g0, L, sweeps):
    """Impurity solver call.
    Takes the propagator :math:`\\mathcal{G}^0(\\tau)` and transforms it
    into a discretized matrix as

    .. math:: \\mathcal{G}^0_{ij} = \\mathcal{G}^0(i\\Delta\\tau - j\\Delta\\tau)

    """
    g0[0] = -g0[lfak]

    gind = lfak + np.arange(lfak).reshape(-1, 1)-np.arange(lfak).reshape(1, -1)
    gx = g0[gind]

    gup = gnewclean(gx, 1.)
    gdw = gnewclean(gx, -1.)

    gstup, gstdw = mcs(sweeps, gup, gdw)

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
    """Returns the interacting function

    .. math:: G'_{ij} = B^{-1}_{ij}G_{ij}
    .. math:: u_j = \\exp(\\sigma v_j) - 1
    .. math:: B_{ij} = \\delta_{ij} - u_j ( G_{ij} - \\delta_{ij}) \\text{no sumation on } j


    """
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
G0w = greenF(w)


v = ising_v(lamb)

for i in range(4):
    G0t = iFFT(G0w)
    g0t = extract_g0t(G0t)
    gt = hf_solver(g0t)
    plt.plot(gt, label='it {}'.format(i))

    Gt = interpol(gt[lfak:])
#    G0t = interpol(g0t[lfak:])

    Gw = FFT(Gt)
#    G0w = FFT(G0t)
#    sigma = dyson_sigma(Gw, G0w)
#
#    Gw = greenF(w, sigma[1::2])
#    G0w = dyson_g0(Gw, sigma)
    G0w = np.zeros(2*Lrang, dtype=np.complex)
    G0w[1::2] = 1/(w - .25*Gw[1::2])


plt.legend(loc=0)
