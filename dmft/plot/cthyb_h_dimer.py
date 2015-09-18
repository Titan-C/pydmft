# -*- coding: utf-8 -*-
"""
Plotting utilities for the dimer in a bethe lattice
===================================================

"""

from __future__ import division, print_function, absolute_import
from glob import glob
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
import re
from dmft.plot.cthyb_h_single_site import show_conv

plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})

def plot_gf(beta, u_int, col, filestr='B{}_U{}/Gf.out.*', n_freq=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True)
    last_file = sorted(glob(filestr.format(beta, u_int)))[-1]
    gf_out = np.loadtxt(last_file).T
    axes[0].plot(gf_out[0], gf_out[1], label='Real')
    axes[0].plot(gf_out[0], gf_out[2], label='Imag')
    axes[1].plot(gf_out[0], gf_out[3], label='Real')
    axes[1].plot(gf_out[0], gf_out[4], label='Imag')
    axes[0].set_xlim([0, xlim])
    axes[1].legend(loc=0, ncol=n_freq)
    graf = r'$G(i\omega_n)$' if 'Gf' in filestr else r'$\Sigma(i\omega_n)$'
    axes[0].set_title('Green functions of the bonding and antibonding bands')
    axes[0].set_ylabel(graf + 'Bond')
    axes[0].set_xlabel(r'$i\omega_n$')
    axes[1].set_title(graf + 'ABond')
    axes[1].set_ylabel(graf + 'ABond')

    plt.show()
    plt.close()
