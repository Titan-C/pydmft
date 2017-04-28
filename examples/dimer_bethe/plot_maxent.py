# -*- coding: utf-8 -*-
r"""
Analytically continue a dimer GF by Maxent
==========================================

Here I continue an insulator
"""
# Created Mon Jan 30 14:13:23 2017
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import dmft.common as gf
from dmft import dimer
from pymaxent.maxent import Maxent
from pymaxent.draw import gf_plot_study
from pymaxent.tools import gaussian, hilbert_trans, differential_weight

# plt.matplotlib.rcParams.update({'axes.labelsize': 22, 'xtick.labelsize':
# 14, 'ytick.labelsize': 14, 'axes.titlesize': 22})

# Output for G_sym= G_AA+G_AB
# G0 = w -tp - (G_sym)
# S = w -tp - t^2 G_sym - G_sym^-1

fac = np.arctan(10 * np.sqrt(3) / 2.5)
omega = np.tan(np.linspace(-fac, fac, 321)) * 2.5 / np.sqrt(3)

tp = 0.1
BETA = 200.
u_int = 2.29
u_str = 'U' + str(u_int)
workdir = "/home/oscar/orlando/dev/dmft-learn/examples/dimer_bethe/tp03f/"
workdir = "/home/oscar/orlando/dev/dmft-learn/examples/dimer_bethe/"

filename = workdir + 'DIMER_PM_{}_B{}_tp{}.h5'.format('ins', BETA, tp)
giw = dimer.extract_flat_gf_iter(filename, u_int, 2)
nfreq = giw.shape[-1]
wn = gf.matsubara_freq(BETA, nfreq, 1 - nfreq)

giw = giw.reshape(-1, 2, 2, nfreq)
giw[:, 1] = -giw[:, 1].conjugate()
giw = giw.reshape(-1, nfreq)
gerr = giw.std(0).clip(3e-4)


defaultM = gaussian(omega, tp, 0.25 + u_int**2 / 4)

Model_gw = Maxent(omega=omega, defaultModel=defaultM, tol=1e-5,
                  minimizer='Bryan', w_n=wn, giw=giw.mean(0),
                  giw_std=gerr, max_nfreq=int(2 * BETA))
Model_gw.getAllSpecFs(alphamin=0.6, alphamax=2.5, numAlpha=24)
gf_plot_study(omega, Model_gw)
plt.show()

# np.savez('AwmetB25tp0.3{}G_sym'.format(u_str), Aw = Model_gw.aveSpecFs, w = Model_gw.omega)

###############################################################################
# How was the Green function fitted
# ---------------------------------
plt.figure('gffit')
gfit = Model_gw.restoreG(Model_gw.aveSpecFs)
wnfit = Model_gw.wn

plt.plot(wn, giw.mean(0).real, 'o:')
plt.plot(wn, giw.mean(0).imag, 'o:')

plt.plot(wnfit, gfit.real, 'x')
plt.plot(wnfit, gfit.imag, 'x')
plt.xlabel(r'$\omega_n$')
plt.title(r'$U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, BETA))
plt.xlim(-5, 5)


plt.figure('gd')
plt.plot(wn, gerr, label='std(G)')
plt.plot(wnfit, np.abs(gfit - giw.mean(0)
                       [int(len(wn) / 2 - BETA):int(len(wn) / 2 + BETA)]), label='$|G-G_{MEM}|$')
plt.legend()
plt.xlabel(r'$\omega_n$')
plt.ylabel('error')
plt.title(r'$U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, BETA))
plt.xlim(-5, 5)

plt.show()
