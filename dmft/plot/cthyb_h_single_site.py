# -*- coding: utf-8 -*-
"""
Plotting utilities for the Single site DMFT phase diagram
=========================================================

"""

from __future__ import division, print_function, absolute_import
from glob import glob
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
import re
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_int, filestr='B{}_U{}/Gf.out.*', col=2, n_freq=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8))
    freq_arr = []
    for step in sorted(glob(filestr.format(beta, u_int))):
        gf_out = np.loadtxt(step).T
        w_n, gf_iw = gf_out[0], gf_out[col]
        axes[0].plot(w_n, gf_iw)
        freq_arr.append(gf_iw[:n_freq])
    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        axes[1].plot(freqs, 'o-.', label=str(num))
    axes[0].set_xlim([0, xlim])
    axes[1].legend(loc=0, ncol=n_freq)
    graf = r'$G(i\omega_n)$' if 'Gf' in filestr else r'$\Sigma(i\omega_n)$'
    axes[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, beta, u_int))
    axes[0].set_ylabel(graf)
    axes[0].set_xlabel(r'$i\omega_n$')
    axes[1].set_title('Evolution of the first frequencies')
    axes[1].set_ylabel(graf+'$(l)$')
    axes[1].set_xlabel('iterations')

    plt.show()
    plt.close()


def list_show_conv(beta, dirstr='B{}_U{}', col=2,
                   n_freq=5, xlim=2, func='/Gf.out.*'):
    """Provides a list of all convergence plots at a given temperature"""
    list_dirs = sorted(glob(dirstr.format(beta, '*')))

    for ldir in list_dirs:
        u_int = float(re.findall(r"U([\d\.]+)", ldir)[0])
        show_conv(beta, u_int, dirstr+func, col, n_freq, xlim)


def averager(vector):
    """Averages over the files terminating with the numbers given in vector"""
    simgiw = 0
    for step in vector:
        w_n, regiw, imgiw = np.loadtxt(step).T
        simgiw += imgiw

    regiw[:] = 0.
    simgiw /= len(vector)
    return np.array([w_n, regiw, simgiw])


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


def phases(dirstr):
    """Scatter plot of the DOS at Fermi level

    Shows the phase diagram of the impurity model of DMFT"""
    for beta in np.array([20., 40., 64., 100., 160.]):
        u_range, gfit, _ = fit_dos(beta, 2, dirstr)
        temp = np.ones(len(u_range)) * 4 / beta
        plt.scatter(u_range, temp, s=300, c=[dos(0) for dos in gfit],
                    vmin=-2, vmax=0)

    plt.xlabel('U/D')
    plt.ylabel('T/t')
