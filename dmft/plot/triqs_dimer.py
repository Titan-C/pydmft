# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq
from pytriqs.plot.mpl_interface import oplot
from pytriqs.archive import HDFArchive
from dmft.plot.hf_single_site import label_convergence
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, tp=0.25, filestr='B{}_U{}.h5', block=2, n_freq=2, xlim=2, skip=0):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    with HDFArchive(filestr.format(tp=tp, BETA=beta), 'r') as datarecord:
        for step in datarecord[u_str].keys()[skip:]:
            labels = [name for name in datarecord[u_str][step]['G_iw'].indices]
            gf_iw = datarecord[u_str][step]['G_iw'][labels[block]]
            axes[0].oplot(gf_iw.imag, 'bo:', label=None)
            axes[0].oplot(gf_iw.real, 'gs:', label=None)
            freq_arr.append([gf_iw.data[:n_freq, 0, 0].real,
                             gf_iw.data[:n_freq, 0, 0].imag])
    freq_arr = np.asarray(freq_arr).T
    for num, (rfreqs, ifreqs) in enumerate(freq_arr):
        axes[1].plot(rfreqs, 's-.', label=str(num))
        axes[1].plot(ifreqs, 'o-.', label=str(num))

    graf = r'$G(i\omega_n)$'
    label_convergence(beta, u_str+'\n$t_\\perp={}$'.format(tp),
                      axes, graf, n_freq, xlim)


def list_show_conv(BETA, tp, filestr='tp{}_B{}.h5', n_freq=5, xlim=2, skip=5):
    """Plots in individual figures for all interactions the DMFT loops"""
    with HDFArchive(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        urange = output_files.keys()

    for u_str in urange:
        show_conv(BETA, u_str, tp, filestr, 2, n_freq, xlim, skip)

        plt.show()
        plt.close()


def phase_diag(BETA, tp_range, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):

    for tp in tp_range:
        tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
        with HDFArchive(filestr.format(tp=tp, BETA=BETA), 'r') as results:
            fl_dos = []
            for u_str in results.keys():
                lastit = results[u_str].keys()[-1]
                labels = [name for name in results[u_str][lastit]['G_iw'].indices]
                gf_iw = results[u_str][lastit]['G_iw'][labels[2]]
                fl_dos.append(gf.fit_gf(w_n[:3], gf_iw.data[:, 0, 0].imag)(0.))

            u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
            plt.scatter(np.ones(len(fl_dos))*tp, u_range, c=fl_dos,
                        s=150, vmin=-2, vmax=0, cmap=plt.get_cmap('inferno'))
    plt.xlim([0, 1])
    plt.title(r'Phase diagram at $\beta={}$'.format(BETA))
    plt.xlabel(r'$t_\perp/D$')
    plt.ylabel('$U/D$')
