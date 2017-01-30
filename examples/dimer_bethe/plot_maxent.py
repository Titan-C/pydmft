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
import h5py
import dmft.common as gf
from pymaxent.maxent import Maxent
from pymaxent.draw import gf_plot_study
from pymaxent.tools import gaussian, hilbert_trans, differential_weight


# plt.matplotlib.rcParams.update({'axes.labelsize': 22, 'xtick.labelsize':
# 14, 'ytick.labelsize': 14, 'axes.titlesize': 22})

# Output for G_sym= G_AA+G_AB
# G0 = w -tp - (G_sym)
# S = w -tp - t^2 G_sym - G_sym^-1


def symdata(gfs_array):
    """Assuming the gfs_array has shape (n_freqs, 4)
conjugate the last 2 columns to make the complete array be SYM"""
    gfs_array[2:] = -gfs_array[2:].conjugate()
    return gfs_array


def make_gf(h5data):
    pos_giw = np.squeeze(h5data.view(np.complex128))
    return np.concatenate((pos_giw[::-1].conjugate(), pos_giw))


def dimer_gf(filename, u_int, iteration):
    with h5py.File(filename, 'r') as datarecord:
        iteration = list(datarecord[u_int])[iteration]
        dat = [make_gf(datarecord[u_int][iteration]['G_iw'][name]['data'].value)
               for name in ['sym_up', 'sym_dw', 'asym_up', 'asym_dw']]
    return np.array(dat)

fac = np.arctan(10 * np.sqrt(3) / 2.5)
omega = np.tan(np.linspace(-fac, fac, 121)) * 2.5 / np.sqrt(3)
defaultM = gaussian(omega, 0.3, 0.25 + 1.8**2 / 4)

u_str = 'U2.15'
tp = 0.3
beta = 25.93
workdir = "/home/oscar/orlando/dev/dmft-learn/examples/dimer_bethe/tp03f/"

filename = workdir + 'DIMER_PM_{}_B{}_tp{}.h5'.format('ins', beta, tp)
giw = symdata(dimer_gf(filename, u_str, -1))

wn = gf.matsubara_freq(beta, giw.shape[-1], 1 - giw.shape[-1])
plt.plot(wn, giw.std(0).clip(3e-4))

Model_gw = Maxent(omega=omega, defaultModel=defaultM, tol=1e-5, std=(True, 1.0),
                  minimizer='Bryan', w_n=wn, giw=giw.mean(0),
                  giw_std=giw.std(0).clip(3e-3), max_nfreq=82)
Model_gw.getAllSpecFs(alphamin=1, alphamax=2.5, numAlpha=25)
gf_plot_study(omega, Model_gw)

np.savez('AwmetB25tp0.3{}G_sym'.format(u_str),
         Aw=Model_gw.aveSpecFs, w=Model_gw.w)

###############################################################################
# How was the Green function fitted
# ---------------------------------

plt.figure('gffit')
gfit = Model_gw.restoreG(Model_gw.aveSpecFs)
wnfit = Model_gw.wn
plt.plot(wnfit, gfit.real, 'o')
plt.plot(wnfit, gfit.imag, 'o')

plt.plot(wn, giw.mean(0).real, 'x:')
plt.plot(wn, giw.mean(0).imag, 'x:')
