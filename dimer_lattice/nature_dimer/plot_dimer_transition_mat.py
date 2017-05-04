# -*- coding: utf-8 -*-
r"""
Dimer Mott transition
=====================

Follow the spectral function from the correlated metal into the dimer
Mott insulator. The spectral functions is decomposed into the bonding
and anti-bonding contributions to make it explicit that is is a
phenomenon of the quasiparticles opening a band gap.

Using Matsubara frequency solver

.. seealso::
    :ref:`sphx_glr_dimer_lattice_nature_dimer_plot_order_param_transition.py`
"""

# author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import dmft.common as gf
from dmft import ipt_imag
import dmft.dimer as dimer


def ipt_u_tp(urange, tp, beta, w):

    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=2**11))
    giw_d, giw_o = dimer.gf_met(w_n, 0., tp, 0.5, 0.)

    w_set = list(np.arange(0, 120, 2))
    w_set = w_set + list(np.arange(120, 512, 8))
    imgss = []

    for u_int in urange:
        giw_d, giw_o, loops = dimer.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-9)
        g0iw_d, g0iw_o = dimer.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt_imag.dimer_sigma(
            u_int, tp, g0iw_d, g0iw_o, tau, w_n)

        ss = gf.pade_continuation(
            1j * siw_d.imag + siw_o.real, w_n, w + 0.0005j, w_set)  # A-bond

        imgss.append(gf.semi_circle_hiltrans(
            w - tp - (ss.real - 1j * np.abs(ss.imag))))
    return imgss


w = np.linspace(-4, 4, 2**12)
BETA = 512.
tp = 0.3
uranget3 = [0.2, 1., 2., 3., 3.45, 3.5]
imgsst3 = ipt_u_tp(uranget3, tp, BETA, w)

tp = 0.8
uranget8 = [0.2, 1., 1.2, 1.35, 2., 3.]
imgsst8 = ipt_u_tp(uranget8, tp, BETA, w)


def plot_dos(urange, imgss, ax, labelx):
    for i, (U, gss) in enumerate(zip(urange, imgss)):
        imgss = -gss.imag
        imgsa = imgss[::-1]
        shift = -2.1 * i
        ax.plot(w, shift + imgss, 'C0', lw=0.5)
        ax.plot(w, shift + imgsa, 'C1', lw=0.5)
        ax.plot(w, shift + (imgss + imgsa) / 2, 'k', lw=2.5)
        ax.axhline(shift, color='k', lw=0.5)
        ax.text(labelx, 1.35 + shift, r"$U={}$".format(U), size=15)
    ax.set_xlabel(r'$\omega$')
    ax.set_xlim([-3.1, 3.1])
    ax.set_ylim([shift, 2.1])
    ax.set_yticks([])


plt.rcParams['figure.autolayout'] = False
fig, (at3, at8) = plt.subplots(1, 2, sharex=True, sharey=True)
plot_dos(uranget3, imgsst3, at3, -3)
plot_dos(uranget8, imgsst8, at8, 1.)
at3.set_title(r'$t_\perp=0.3$')
at8.set_title(r'$t_\perp=0.8$')
plt.subplots_adjust(wspace=0.02)
plt.savefig('dimer_transition_spectra.pdf')
plt.close()
