# -*- coding: utf-8 -*-
r"""
====================
Study of self energy
====================

"""
# Created Wed Apr  6 08:40:36 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import dmft.plot.triqs_dimer as tdp
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.plot.hf_single_site
import dmft.ipt_imag as ipt


def pade_diag(gf_s, gf_a, w_n, w_set, w):
    pc = gf.pade_coefficients(gf_s[w_set], w_n[w_set])
    gr_s = gf.pade_rec(pc, w + 5e-8j, w_n[w_set])

    pc = gf.pade_coefficients(gf_a[w_set], w_n[w_set])
    gr_a = gf.pade_rec(pc, w + 5e-8j, w_n[w_set])

    return gr_s, gr_a


def plot_functions(w, ss_w, sa_w, U, mu, tp, beta, plot_second):
    f, ax = plt.subplots(1, sharex=True)
    ax.plot(w, ss_w.real, label='Re sum')
    ax.plot(w, -ss_w.imag, label='-Im sum')
    if plot_second:
        ax.plot(w, sa_w.real, label='Re dif')
        ax.plot(w, -sa_w.imag, label='-Im dif')
    ax.legend(loc=0)
    ax.set_xlabel(r'$\omega$')
    ax.set_title(
        r'Isolated dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

tp = 0.3
U = 3.5
with tdp.HDFArchive('tp03f/DIMER_PM_B100.0_tp0.3.h5') as data:
    giw = tdp.get_giw(data['U' + str(U)], slice(-1, -4, -1))

giw_s = np.squeeze(.5 * (giw['sym_up'].data + giw['sym_dw'].data))
giw_s = np.squeeze(giw['sym_up'].data)[len(giw_s) / 2:len(giw_s) / 2 + 300]

giw_as = np.squeeze(.5 * (giw['asym_up'].data + giw['asym_dw'].data))
giw_as = np.squeeze(giw['asym_up'].data)[
    len(giw_as) / 2:len(giw_as) / 2 + 300]


w_n = gf.matsubara_freq(100., 300)
w = np.linspace(-5, 20, 1000)
w_set = np.arange(100)
gs, ga = pade_diag(giw_s, giw_as, w_n, w_set, w)
plot_functions(w, gs, ga, U, 0, tp, 100., False)
plt.ylabel(r'$G(\omega)$')
plt.ylim([-.5, 2])
siw_s = 1j * w_n - tp - .25 * giw_s - 1 / giw_s
siw_as = 1j * w_n + tp - .25 * giw_as - 1 / giw_as

ss, sa = pade_diag(siw_s, siw_as, w_n, w_set, w)
plot_functions(w, ss, sa, U, 0, tp, 100., False)
plt.ylabel(r'$\Sigma(\omega)$')
plt.ylim([-5, 5])

gs, ga = gf.greenF(-1j * w, + tp + ss), gf.greenF(-1j * w, - tp + sa)
plot_functions(w, gs, ga, U, 0, tp, 100., False)
plt.ylabel(r'$G(\omega)$')
plt.ylim([-.5, 2])

tp = 0.3
U = 2.1499
with tdp.HDFArchive('tp03f/DIMER_PM_B100.0_tp0.3.h5') as data:
    giw = tdp.get_giw(data['U' + str(U)], slice(-1, -3, -1))

giw_s = np.squeeze(.5 * (giw['sym_up'].data + giw['sym_dw'].data))
#giw_s = np.squeeze(giw['sym_up'].data)
giw_s = np.squeeze(giw['sym_up'].data)[len(giw_s) / 2:len(giw_s) / 2 + 300]


giw_as = np.squeeze(.5 * (giw['asym_up'].data + giw['asym_dw'].data))
#giw_as = np.squeeze(giw['asym_up'].data)
giw_as = np.squeeze(giw['asym_up'].data)[
    len(giw_as) / 2:len(giw_as) / 2 + 300]

w_n = gf.matsubara_freq(100., 300)
w = np.linspace(-5, 5, 1000)
w_set = np.arange(80)
gs, ga = pade_diag(giw_s, 0.5 * (giw_s - giw_as.real +
                                 1j * giw_as.imag), w_n, w_set, w)
plot_functions(w, gs, ga, U, 0, tp, 100., False)
plt.ylabel(r'$G(\omega)$')
plot_functions(w, ga, gs, U, 0, tp, 100., False)
plt.ylabel(r'$G(\omega)$')
