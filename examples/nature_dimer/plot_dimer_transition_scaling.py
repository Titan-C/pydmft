# -*- coding: utf-8 -*-
r"""
Dimer Mott transition Scaled quantities
=======================================

Follow the spectral function from the correlated metal into the dimer
Mott insulator. The spectral functions is decomposed into the bonding
and anti-bonding contributions to make it explicit that is is a
phenomenon of the quasiparticles opening a band gap.

.. seealso::
    :ref:`sphx_glr_auto_examples_nature_dimer_plot_order_param_transition.py`
    :ref:`sphx_glr_auto_examples_nature_dimer_plot_dimer_transition.py`
"""

# author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import dmft.common as gf
import dmft.ipt_real as ipt

plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22,
                                'mathtext.fontset': 'cm'})

w = np.linspace(-4, 4, 2**12)
dw = w[1] - w[0]

beta = 800.
nfp = gf.fermi_dist(w, beta)


def low_en_qp(ss):
    glp = np.array([0.])
    sigtck = splrep(w, ss.real, s=0)
    sig_0 = splev(glp, sigtck, der=0)[0]
    dw_sig0 = splev(glp, sigtck, der=1)[0]
    quas_z = 1 / (1 - dw_sig0)
    return quas_z, sig_0, dw_sig0

###############################################################################
# The :math:`t_\perp/D=0.3` scenario
# ==================================
#


tp = 0.3
gss = gf.semi_circle_hiltrans(w + 5e-3j - tp)
gsa = gf.semi_circle_hiltrans(w + 5e-3j + tp)
urange = np.arange(0.2, 3.3, 0.3)
urange = [0.2, 1., 2., 3., 3.47]
plt.close('all')


def plot_row(gss, gsa, ss, ax, i):
    quas_z, sig_0, dw_sig0 = low_en_qp(ss)
    tpp = (tp + sig_0) * quas_z
    llg = gf.semi_circle_hiltrans(w + 1e-8j - tpp, quas_z) * quas_z

    shift = -2.1 * i
    ax.plot(w / quas_z, shift + -gss.imag, 'C0')
    ax.plot(w / quas_z, shift + -gsa.imag, 'C1')
    ax.plot(w / quas_z, shift + -(gss + gsa).imag / 2, 'k', lw=2)
    ax.plot(w / quas_z, shift + -llg.imag, "C3--", lw=2)
    ax.axhline(shift, color='k', lw=0.5)
    return shift


ax = plt.subplot(111)
for i, U in enumerate(urange):
    (gss, gsa), (ss, sa) = ipt.dimer_dmft(
        U, tp, nfp, w, dw, gss, gsa, conv=1e-4)
    shift = plot_row(gss, gsa, ss, ax, i)


plt.xlabel(r'$\omega/ZD$')
plt.xlim([-4, 4])
plt.ylim([shift, 2.1])
plt.yticks(0.5 - 2.1 * np.arange(len(urange)), ['U=' + str(u) for u in urange])
# plt.savefig('dimer_transition_spectra_scaling.pdf')

###############################################################################
# The :math:`t_\perp/D=0.8` scenario
# ==================================
#
tp = 0.8
gss = gf.semi_circle_hiltrans(w + 5e-3j - tp)
gsa = gf.semi_circle_hiltrans(w + 5e-3j + tp)
urange = np.linspace(0.2, 1.64, 6)
urange = [0.5, 1., 1.2, 1.352, 1.4]
plt.close('all')
ax = plt.subplot(111)
for i, U in enumerate(urange):
    (gss, gsa), (ss, sa) = ipt.dimer_dmft(
        U, tp, nfp, w, dw, gss, gsa, conv=1e-4)
    shift = plot_row(gss, gsa, ss, ax, i)

plt.xlabel(r'$\omega$')
plt.xlim([-4, 4])
plt.ylim([shift, 2.1])
plt.yticks(0.5 - 2.1 * np.arange(len(urange)), ['U=' + str(u) for u in urange])

# plt.savefig('dimer_transition_spectra_tp0.8_scaling.pdf')
plt.close('all')
