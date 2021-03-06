# -*- coding: utf-8 -*-
r"""
=========================================
Studying the scattering rate in the dimer
=========================================

Do the zero frequency extrapolation of the Self-Energy to find the
scattering rate in the metal and the renormalized intra site hopping

"""
# Created Mon May  9 11:30:14 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import dmft.common as gf
import py3qs.triqs_dimer as tdp

plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 18, 'ytick.labelsize': 18,
                                'axes.titlesize': 22})
os.chdir('/home/oscar/orlando/dev/dmft-learn/examples/dimer_bethe/tp03f/')


def processing(BETA, U, tp, ax):
    with tdp.HDFArchive('DIMER_PM_B{}_tp0.3.h5'.format(BETA)) as data:
        # print(data)
        giw = tdp.get_giw(data['U' + str(U)], slice(-1, -3, -1))

    giw_s = np.squeeze(.5 * (giw['sym_up'].data + giw['sym_dw'].data))
    giw_s = giw_s[len(giw_s) / 2:len(giw_s) / 2 + 300]
    w_n = gf.matsubara_freq(BETA, len(giw_s))

    giw_a = np.squeeze(.5 * (giw['asym_up'].data + giw['asym_dw'].data))
    giw_a = giw_a[len(giw_a) / 2:len(giw_a) / 2 + 300]

    # more Avgs
    giw_s = 0.5j * (giw_s + giw_a).imag + 0.5 * (giw_s - giw_a).real

    x = int(.9 * BETA)
    der1 = (giw_s.real * w_n**2)[x:int(1.2 * BETA)].mean()
    der2 = ((giw_s.imag + 1 / w_n) * w_n**3)[x:int(1.2 * BETA)].mean()
    tails = -1j / w_n + der1 / w_n**2 + der2 * 1j / w_n**3
    giw_s[x:] = tails[x:]

    sidab = 1j * w_n - tp - .25 * giw_s - 1 / giw_s
    w = np.linspace(0, 1, 40)
    swa = gf.pade_continuation(sidab, w_n[:int(1.8 * BETA)], 1j * w)
    ax[0].plot(w_n, sidab.imag)
    ax[0].plot(w, swa.imag, "k:")
    ax[1].plot(w_n, sidab.real)
    ax[1].plot(w, swa.real, "k:")

    sig_11_0 = np.polyfit(w_n[:2], sidab.imag[:2], 1)[1]
    rtp = np.polyfit(w_n[:2], sidab.real[:2], 1)[1]

    return swa.imag[0], swa.real[0], sig_11_0, rtp

###############################################################################
# Fit of the Self-Energy by Padé and linear fit from the first 2 frequencies
# Work on metallic case at U=2.5, tp=0.3

BETARANGE = np.array([20., 24., 30., 40., 50., 62., 75., 100., 150.])
fig, si = plt.subplots(2, 1, sharex=True)
zero_wp = np.array([processing(BETA, 2.15, .3, si) for BETA in BETARANGE])
si[0].set_ylabel(r'$\Im m \Sigma_{11}$')
si[1].set_ylabel(r'$\Re e\Sigma_{12}$')
si[1].set_xlabel(r'$i\omega_n$')
si[1].set_xlim([0, 1])
si[1].set_ylim([0.1, 0.26])
#fig.savefig('CT_QMC-Sigma_zero_w.pdf', format='pdf', transparent=False, bbox_inches='tight', pad_inches=0.05)

###############################################################################
# Blues are by Padé, Greens are linear. Not to much difference

fig, si = plt.subplots(2, 1, sharex=True)
si[0].plot(1 / BETARANGE, - zero_wp[:, 0])
si[0].plot(1 / BETARANGE, - zero_wp[:, 2])
si[1].plot(1 / BETARANGE, .3 + zero_wp[:, 1])
si[1].plot(1 / BETARANGE, .3 + zero_wp[:, 3])
si[0].set_ylabel(r'$-\Im m \Sigma_{11}(w=0)$')
si[0].grid()
si[1].set_ylabel(r'$t_\perp + \Re e\Sigma_{12}(w=0)$')
si[1].set_xlabel('$T/D$')
si[1].grid()
#fig.savefig('CT_QMC-Sigma_scattering.pdf', format='pdf', transparent=False, bbox_inches='tight', pad_inches=0.05)
