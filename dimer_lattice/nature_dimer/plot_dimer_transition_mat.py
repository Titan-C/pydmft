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
###############################################################################
# The :math:`t_\perp/D=0.3` scenario
# ==================================
#
BETA = 512.
tp = 0.3
urange = [0.2, 1., 2., 3., 3.45, 3.5]
imgss = ipt_u_tp(urange, tp, BETA, w)
plt.close('all')
for i, (U, gss) in enumerate(zip(urange, imgss)):
    imgss = -gss.imag
    imgsa = imgss[::-1]
    shift = -2.1 * i
    plt.plot(w, shift + imgss, 'C0', lw=0.5)
    plt.plot(w, shift + imgsa, 'C1', lw=0.5)
    plt.plot(w, shift + (imgss + imgsa) / 2, 'k', lw=2.5)
    plt.axhline(shift, color='k', lw=0.5)
    plt.text(-2.8, 1.45 + shift, r"$U/D={}$".format(U), size=16)
plt.xlabel(r'$\omega$')
plt.xlim([-3, 3])
plt.ylim([shift, 2.1])
plt.yticks([])
# plt.savefig('dimer_transition_spectra.pdf')

###############################################################################
# The :math:`t_\perp/D=0.8` scenario
# ==================================
#
tp = 0.8
urange = [0.2, 1., 1.2, 1.35, 2., 3.]
imgsst8 = ipt_u_tp(urange, tp, BETA, w)
plt.close('all')
for i, (U, gss) in enumerate(zip(urange, imgsst8)):
    imgss = -gss.imag
    imgsa = imgss[::-1]
    shift = -2.1 * i
    plt.plot(w, shift + imgss, 'C0', lw=0.5)
    plt.plot(w, shift + imgsa, 'C1', lw=0.5)
    plt.plot(w, shift + (imgss + imgsa) / 2, 'k', lw=2.5)
    plt.axhline(shift, color='k', lw=0.5)
    plt.text(1.8, 1.45 + shift, r"$U/D={}$".format(U), size=16)
plt.xlabel(r'$\omega$')
plt.xlim([-3, 3])
plt.ylim([shift, 2.1])
plt.yticks([])
# plt.savefig('dimer_transition_spectra_tp0.8.pdf')
