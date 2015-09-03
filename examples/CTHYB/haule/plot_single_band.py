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
import matplotlib.pyplot as plt
import numpy as np
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
    graf = r'$G(i\omega)$' if 'Gf' in filestr else r'$\Sigma(i\omega_n)$'
    ax[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, beta, U))
    ax[1].set_title('Evolution of the first frequencies')
    plt.show()
    plt.close()


###############################################################################
# I start first by checking the convergence of the system at various
# data points for this I have a look at the evolulution of the
# outputed Green's functions and the self energy on every loop
# performing as many iteration to see the system stabilize

show_conv(64., 2.4, 'coex/B{}_U{}/Sig.out.*')

###############################################################################
# This first plot demostrates that for the simply metalic state the
# system is quite well behaved and the convergence is quite
# simple. Few iterations are necessary but then there always remains
# the monte carlo noise in the solution.
