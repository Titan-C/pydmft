# -*- coding: utf-8 -*-
r"""
Dimer Mott transition
=====================

Follow the spectral function from the correlated metal into the dimer
Mott insulator. The spectral functions is decomposed into the bonding
and anti-bonding contributions to make it explicit that is is a
phenomenon of the quasiparticles opening a band gap.

.. seealso::
    :ref:`sphx_glr_auto_examples_RKKY_plot_order_param_transition.py`
"""

# Created Wed Feb 15 19:19:47 2017

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import dmft.common as gf
import dmft.ipt_real as ipt

from slaveparticles.quantum.operators import fermi_dist

plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22,
                                'mathtext.fontset': 'cm'})

w = np.linspace(-4, 4, 2**12)
dw = w[1] - w[0]

beta = 800.
nfp = fermi_dist(w, beta)

tp = 0.3
gss = gf.semi_circle_hiltrans(w + 5e-3j - tp)
gsa = gf.semi_circle_hiltrans(w + 5e-3j + tp)
urange = np.arange(0.2, 3.3, 0.3)
urange = [0.2, 1., 2., 3., 3.47, 3.5]
plt.close('all')
for i, U in enumerate(urange):
    (gss, gsa), (ss, sa) = ipt.dimer_dmft(
        U, tp, nfp, w, dw, gss, gsa, conv=1e-4)

    plt.gca().set_prop_cycle(None)
    shift = -2.1 * i
    plt.plot(w, shift + -gss.imag)
    plt.plot(w, shift + -gsa.imag)
    plt.plot(w, shift + -(gss + gsa).imag / 2, 'k', lw=2)
    plt.axhline(shift, color='k', lw=0.5)

plt.xlabel(r'$\omega$')
plt.xlim([-4, 4])
plt.ylim([shift, 2.1])
plt.yticks(0.5 - 2.1 * np.arange(len(urange)), ['U=' + str(u) for u in urange])
# plt.savefig('dimer_transition_spectra.pdf')
