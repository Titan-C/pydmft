# -*- coding: utf-8 -*-
"""
Analyzing the single band single site Metal-Insulator Transition
================================================================

Reviewing the single band case and expecting to find the famous
coexistence region I start looping at many different points in the
phase diagram. The iterate.py script is this same folder is reponsible
for creating the data but it still requires manual input for the
searched data point.
"""

from __future__ import division, print_function, absolute_import
from glob import glob
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
import os
import re
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, U, filestr='B{}_U{}/Gf.out.*', nf=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    f, ax = plt.subplots(1, 2, figsize=(13, 8))
    freq_arr = []
    for it in sorted(glob(filestr.format(beta, U))):
        wn, regiw, imgiw = np.loadtxt(it).T
        ax[0].plot(wn, imgiw)
        freq_arr.append(imgiw[:nf])
    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        ax[1].plot(freqs, 'o-.', label=str(num))
    ax[0].set_xlim([0, 2])
    ax[1].legend(loc=0, ncol=nf)
    graf = r'$G(i\omega_n)$' if 'Gf' in filestr else r'$\Sigma(i\omega_n)$'
    ax[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, beta, U))
    ax[0].set_ylabel(graf)
    ax[0].set_xlabel(r'$i\omega_n$')
    ax[1].set_title('Evolution of the first frequencies')
    ax[1].set_ylabel(graf+'$(l)$')
    ax[1].set_xlabel('iterations')

    plt.show()
    plt.close()


def list_show_conv(beta, dirstr='B{}_U{}', nf=5, xlim=2):
    list_dirs = sorted(glob(dirstr.format(beta, '*')))

    for ldir in list_dirs:
        U = float(re.findall("U([\d\.]+)", ldir)[0])
        show_conv(beta, U, dirstr+'/Gf.out.*', nf, xlim)


###############################################################################
# I start first by checking the convergence of the system at various
# data points for this I have a look at the evolulution of the
# outputed Green's functions and the self energy on every loop
# performing as many iteration to see the system stabilize

#show_conv(64., 2.4, 'coex/B{}_U{}/Sig.out.*')

###############################################################################
# This first plot demostrates that for the simply metalic state the
# system is quite well behaved and the convergence is quite
# simple. Few iterations are necessary but then there always remains
# the monte carlo noise in the solution.


def averager(vector):
    """Averages over the files terminating with the numbers given in vector"""
    simgiw = 0
    for it in vector:
        wn, regiw, imgiw = np.loadtxt(it).T
        simgiw += imgiw

    regiw[:] = 0.
    simgiw /= len(vector)
    return np.array([wn, regiw, imgiw])


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
    U = []
    figiw = []
    ligiw = []
    w_n = gf.matsubara_freq(beta, 3)

    for ldir in list_dirs:
        iterations = sorted(glob(ldir + '/Gf.out.*'))[-avg:]
        if not iterations:
            continue

        wn, rgiw, igiw = averager(iterations)
        U.append(float(re.findall("U([\d\.]+)", ldir)[0]))
        gfit = gf.fit_gf(w_n, igiw)
        figiw.append(gfit)
        ligiw.append(igiw)

    return np.asarray(U), figiw, ligiw


def plot_fit_dos(beta, avg, filestr='B{}_U{}/Gf.out.*', xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    f, ax = plt.subplots(1, 2, figsize=(13, 8))

    U, fit_gf, lgiw = fit_dos(beta, avg)
    wn = gf.matsubara_freq(beta)
    wr = np.arange(0, wn[2], 0.05)
    for u, gfit, igiw in zip(U, fit_gf, lgiw):
        ax[0].plot(wn, igiw, 'o:', label='U='+str(u))
        ax[0].plot(wr, gfit(wr), 'k:')

    ax[0].set_xlim([0, xlim])
    ax[1].plot(U, [dos(0) for dos in fit_gf], 'o-')

    plt.show()
    plt.close()


def phases(dirstr):
    fig, axs = plt.subplots()
    for beta in np.array([20., 40., 64., 100.]):
        U, gfit, _ = fit_dos(beta, 2, dirstr)
        T = np.ones(len(U)) * 4 / beta
        axs.scatter(U, T, s=300, c=[dos(0) for dos in gfit],
                    vmin=-2, vmax=0)

    plt.xlabel('U/D')
    plt.ylabel('T/t')
