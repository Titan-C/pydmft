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
from scipy.linalg.blas import dger
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dmft.common import gt_fouriertrans, gw_invfouriertrans, greenF


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


def ising_v(L=32, polar=0.5):
    """initialize the vector V of Ising fields

    .. math:: V = (\\sigma_1, \\sigma_2, \\cdots, \\sigma_L)

    where the vector entries :math:`\\sigma_n=\\pm 1` are randomized subject
    to a threshold given by polar

    Parameters
    ----------
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
    vis[rand > polar] = -1
    return vis


def hf_solver(g0, v, sweeps):
    """Impurity solver call.
    Takes the propagator :math:`\\mathcal{G}^0(\\tau)` and transforms it
    into a discretized matrix as

    .. math:: \\mathcal{G}^0_{ij} = \\mathcal{G}^0(i\\Delta\\tau - j\\Delta\\tau)

    Then it creates the interacting Green functions as given by the contribution
    of the auxiliary discretized spin field.

    """
    lfak = v.size

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
    xg = np.zeros(lfak+1)
    xg[1:-1] = (xga[lfak+1:-1]-xga[1:lfak]) / lfak
    xg[0] = xga[lfak] / lfak
    xg[-1] = 1-xg[0]

    return xg


def mcs(sweeps, gup, gdw, v):
    lfak = v.size
    gstup, gstdw = np.zeros((lfak, lfak)), np.zeros((lfak, lfak))
    kroneker = np.eye(lfak)

    for mcs in range(sweeps):
        for j in range(lfak):
            dv = 2.*v[j]
            ratup = 1. + (1. - gup[j, j])*(np.exp(-dv)-1.)
            ratdw = 1. + (1. - gdw[j, j])*(np.exp( dv)-1.)
            rat = ratup * ratdw
            rat = rat/(1.+rat)
            if rat > np.random.rand():
                v[j] *= -1.
                gup = gnew(gup, v, j, 1., kroneker)
                gdw = gnew(gdw, v, j, -1., kroneker)

        gstup += gup
        gstdw += gdw

    gstup = gstup/sweeps
    gstdw = gstdw/sweeps

    return gstup, gstdw


def gnewclean(gx, v, sign):
    """Returns the interacting function :math:`G_{ij}`

    .. math:: G'_{ij} = B^{-1}_{ij}G_{ij}

    where

    .. math::
        u_j &= \\exp(\\sigma v_j) - 1 \\\\
        B_{ij} &= \\delta_{ij} - u_j ( G_{ij} - \\delta_{ij}) \\text{  no sumation on } j

    """
    ee = np.exp(sign*v) - 1.
    ide = np.eye(v.size)
    b = ide - ee * (gx-ide)

    return solve(b, gx)


def gnew(g, v, k, sign, kroneker):
    """Quick update of green function matrix after a single spin flip of
    the auxiliary field. It calculates

    .. math:: \\alpha = \\frac{\\exp(2\\sigma v_j) - 1}
                        {1 + (1 - G_{jj})(\\exp(2\\sigma v_j) - 1)}
    .. math:: G'_{ij} = G_{ij} + \\alpha (G_{ik} - \\delta_{ik})G_{kj}

    no sumation in the indexes"""
    dv = sign*v[k]*2
    ee = np.exp(dv)-1.
    a = ee/(1. + (1.-g[k, k])*ee)
    x = g[:, k] - kroneker[:, k]
    y = g[k, :]

    return dger(a,x,y,1,1,g,1,1,1)




def extract_g0t(g0t, lfak=32):
    """Extract a reducted amout of points of g0t"""
    gt = interpol(g0t, lfak)

    return np.concatenate((-gt[:-1], gt))


def interpol(gt, Lrang):
    """This function interpolates :math:`G(\\tau)` to an even numbered anti
    periodic array for it to be directly used by the fourier transform into
    matsubara frequencies"""
    t = np.linspace(0, 1, gt.size)
    f = interp1d(t, gt)
    tf = np.linspace(0, 1, Lrang+1)
    return f(tf)


class HF_imp(object):
    """Hirsch and Fye impurity solver in paramagnetic scenario"""
    def __init__(self, dtau=0.5, n_tau=32, lrang=1000):
        """sets up the environment"""
        self.dtau = dtau
        self.n_tau = n_tau
        self.beta = dtau * n_tau
        self.lrang = lrang

    def dmft_loop(self, U=2., mu=0.0, loops=8, mcs=5000):
        """Implementation of the solver"""
        i_omega = 1j*np.pi*(1+2*np.arange(self.beta*6*U)) / self.beta
        fine_tau = np.linspace(0, self.beta, self.lrang + 1)
        G0iw = greenF(i_omega, mu=mu)[1::2]
        v_aux = np.arccosh(np.exp(self.dtau*U/2)) * ising_v(self.n_tau)
        simulation = []
        for i in range(loops):
            G0t = gw_invfouriertrans(G0iw, fine_tau, i_omega, self.beta)
            g0t = extract_g0t(G0t, self.n_tau)

            gt = hf_solver(-g0t, v_aux, mcs)

            Gt = interpol(-gt, self.lrang)
            Giw = gt_fouriertrans(Gt, fine_tau, i_omega, self.beta)
            G0iw = 1/(i_omega + mu - .25*Giw)
            simulation.append({ 'G0iw'  : G0iw,
                                'Giw'   : Giw,
                                'gtau'  : gt})
        return simulation


hf_sol = HF_imp()
import timeit
start_time = timeit.default_timer()

sim = hf_sol.dmft_loop(loops=4)
print(timeit.default_timer() - start_time)
for it, res in enumerate(sim):
    plt.plot(res['gtau'], label='iteration {}'.format(it))
plt.legend(loc=0)
plt.figure()
for it, res in enumerate(sim):
    plt.plot(res['iwn'].imag, res['G0iw'].real, '*-', label='iter Re {}'.format(it))
    plt.plot(res['iwn'].imag, res['Giw'].real, '*-', label='iter Im {}'.format(it))
plt.legend(loc=0)
plt.figure()
for it, res in enumerate(sim):
    sig=1/res['G0iw'] - 1/res['Giw']
    plt.plot(res['iwn'].imag, sig.real, '*-', label='iter Re {}'.format(it))
    plt.plot(res['iwn'].imag, sig.imag, '*-', label='iter Im {}'.format(it))
plt.legend(loc=0)
