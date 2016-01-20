# -*- coding: utf-8 -*-
"""
Plotting utilities for the Single site DMFT phase diagram
=========================================================

"""

from __future__ import division, print_function, absolute_import
from glob import glob
from dmft.plot.hf_single_site import label_convergence
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
import re
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_int, filestr='coex/B{}_U{}/Gf.out.*.npy', skip=0, n_freq=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    w_n = gf.matsubara_freq(beta, 3*beta)
    files = sorted(glob(filestr.format(beta, u_int)))[skip:]
    for step in files:
        giw = np.squeeze(np.load(step))
        if len(giw.shape) > 1:
            axes[0].plot(w_n, giw[0].real, 'gs:', w_n, giw[0].imag, 'bo:')
            freq_arr.append(np.array([giw[0].real[:n_freq], giw[0].imag[:n_freq]]))
        else:
            axes[0].plot(w_n, giw.real, 'gs:', w_n, giw.imag, 'bo:')
            freq_arr.append(giw.imag[:n_freq])
    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        axes[1].plot(freqs.T, 'o-.', label=str(num))

    graf = r'$G(i\omega_n)$' if 'Gf' in filestr else r'$\Sigma(i\omega_n)$'
    label_convergence(beta, 'U'+str(u_int), axes, graf, n_freq, xlim)



def list_show_conv(beta, dirstr='B{}_U{}', col=2,
                   n_freq=5, xlim=2, func='/Gf.out.*'):
    """Provides a list of all convergence plots at a given temperature"""
    list_dirs = sorted(glob(dirstr.format(beta, '*')))

    for ldir in list_dirs:
        u_int = float(re.findall(r"U([\d\.]+)", ldir)[0])
        show_conv(beta, u_int, dirstr+func, col, n_freq, xlim)
        plt.show()
        plt.close()


def averager(vector):
    """Averages over the files terminating with the numbers given in vector"""
    simgiw = 0
    for step in vector:
        simgiw += np.load(step)

    simgiw /= len(vector)

    return simgiw


def fit_dos(beta, avg, dirstr='coex/B{}_U{}'):
    """Fits for all Green's functions at a given beta their
    density of states at the Fermi energy

    Parameters
    ----------
    beta: float inverse temperature
    avg: int number of last iterations to average over

    Return
    ------
    arrays of Interaction, Fitted and source GF imaginary part
    """

    list_dirs = sorted(glob(dirstr.format(beta, '*')))
    u_range = []
    figiw = []
    ligiw = []
    w_n = gf.matsubara_freq(beta, 3)

    for ldir in list_dirs:
        iterations = sorted(glob(ldir + '/Gf.out.*'))[-avg:]
        if not iterations:
            continue

        _, _, igiw = averager(iterations)
        u_range.append(float(re.findall(r"U([\d\.]+)", ldir)[0]))
        gfit = gf.fit_gf(w_n, igiw)
        figiw.append(gfit)
        ligiw.append(igiw)

    return np.asarray(u_range), figiw, ligiw


def plot_fit_dos(beta, avg, filestr='B{}_U{}/Gf.out.*', xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8))

    u_range, fit_gf, lgiw = fit_dos(beta, avg, filestr)
    w_n = gf.matsubara_freq(beta)
    omega = np.arange(0, w_n[2], 0.05)
    for u_int, gfit, igiw in zip(u_range, fit_gf, lgiw):
        axes[0].plot(w_n, igiw, 'o:', label='U='+str(u_int))
        axes[0].plot(omega, gfit(omega), 'k:')

    axes[0].set_xlim([0, xlim])
    axes[1].plot(u_range, [dos(0) for dos in fit_gf], 'o-')

    plt.show()
    plt.close()


def phases(beta_array):
    """Scatter plot of the DOS at Fermi level

    Shows the phase diagram of the impurity model of DMFT"""
    for beta in beta_array:
        u_range, gfit, _ = fit_dos(beta, 2)
        temp = np.ones(len(u_range)) / beta
        plt.scatter(u_range, temp, s=300, c=[dos(0) for dos in gfit],
                    vmin=-2, vmax=0)

    plt.xlabel('U/D')
    plt.ylabel('T/t')
