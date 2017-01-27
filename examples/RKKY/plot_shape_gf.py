# -*- coding: utf-8 -*-
r"""
============================
Form of the Greens Functions
============================

"""
# Author: Óscar Nájera

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np

import dmft.common as gf
import dmft.ipt_imag as ipt
from dmft.ipt_real import dimer_dmft as dimer_dmft_real
import dmft.dimer as dimer
from slaveparticles.quantum import dos


def ipt_u_tp(u_int, tp, beta, seed='ins'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=1024))
    giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'ins':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_d, giw_o, loops = dimer.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-12)
    g0iw_d, g0iw_o = dimer.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    return giw_d, giw_o, siw_d, siw_o, g0iw_d, g0iw_o, w_n


###############################################################################
# Insulator
# ---------
#
u_int = 3.5
BETA = 100.
tp = 0.3
title = r'IPT lattice dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(
    u_int, tp, BETA)
giw_d, giw_o, siw_d, siw_o, g0iw_d, g0iw_o, w_n = ipt_u_tp(
    u_int, tp, BETA, 'ins')

plt.plot(w_n, giw_o.real, 's-')
plt.plot(w_n, giw_d.imag, 'o-')
plt.xlim([0, 5])
plt.close('all')
plt.plot(w_n, g0iw_o.real, 's-')
plt.plot(w_n, g0iw_d.imag, 'o-')
g0sym = 1 / (1j * w_n - tp - .25 * (giw_o + giw_d))
plt.plot(w_n, g0sym.real, '+-')
plt.plot(w_n, g0sym.imag, 'x-')
g0asym = 1 / (1j * w_n + tp - .25 * (-giw_o + giw_d))
plt.plot(w_n, g0asym.real, '+-')
plt.plot(w_n, g0asym.imag, 'x-')
plt.xlim([0, 5])
