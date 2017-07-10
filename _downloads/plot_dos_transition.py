# -*- coding: utf-8 -*-
r"""
===========================================
Evolution of DOS as function of temperature
===========================================

Using a real frequency solver in the IPT scheme the Density of states
is tracked through the first orders transition.
"""
# Created Tue Jun 14 15:44:38 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.signal as signal
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import dmft.common as gf
import dmft.ipt_real as ipt
from dmft.utils import optical_conductivity

from slaveparticles.quantum.operators import fermi_dist
import slaveparticles.quantum.dos as dos

plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22})


def loop_bandwidth(simval, beta, seed='mott gap'):
    """Solves IPT dimer and return Im Sigma_AA, Re Simga_AB

    returns list len(betarange) x 2 Sigma arrays
"""

    s = []
    g = []
    w = np.linspace(-4, 4, 2**12)
    dw = w[1] - w[0]
    gss = gf.semi_circle_hiltrans(w + 5e-3j - 1)
    gsa = gf.semi_circle_hiltrans(w + 5e-3j + 1)
    nfp = dos.fermi_dist(w, beta)
    for D, tp in simval:
        print('D: ', D, 'tp/U: ', tp, 'Beta', beta)
        (gss, gsa), (ss, sa) = ipt.dimer_dmft(
            1, tp, nfp, w, dw, gss, gsa, conv=1e-4, t=(D / 2))
        g.append((gss, gsa))
        s.append((ss, sa))

    return np.array(g), np.array(s), w, nfp


def plot_spectralfunc(gwi, simval, rf, yshift=False):
    shift = 0
    for (gss, gsa), (D, tp), r in zip(gwi, simval, rf):
        Awloc = -.5 * (gss + gsa).imag / np.pi
        print(trapz(Awloc, w))
        plt.plot(w, r * Awloc, label=r'D/U={:.2},tp={}'.format(D, tp))
        #plt.plot(w,  Awloc * nfp, 'k:')

    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$A(\omega)$')
    plt.legend(loc=0)
    plt.xlim([-2, 0])

plt.figure()
simvals = [(1 / 3., .1), (1 / 3., .12), (1 / 2.7, .12)]
giw, swi, w, nfp = loop_bandwidth(simvals, 250)
renorm = [1. / .88, 1 / .85, 1 / .82]
plot_spectralfunc(giw, simvals, renorm)
