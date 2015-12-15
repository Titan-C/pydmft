# -*- coding: utf-8 -*-
from dmft.plot.hf_single_site import label_convergence
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse, GfReFreq, \
    TailGf
from pytriqs.plot.mpl_interface import oplot
import dmft.common as gf
import dmft.plot.cthyb_h_single_site as pss
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, filestr='CH_sb_b{BETA}.h5', n_freq=2, xlim=2, skip=0):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    with HDFArchive(filestr.format(BETA=beta), 'r') as datarecord:
        for step in datarecord[u_str].keys()[skip:]:
            gf_iw = datarecord[u_str][step]['giw']
            gf_iw = .5*(gf_iw['up']+gf_iw['down'])
            axes[0].oplot(gf_iw.imag, 'bo:', label=None)
            axes[0].oplot(gf_iw.real, 'gs:', label=None)
            freq_arr.append([gf_iw.data[:n_freq, 0, 0].real,
                             gf_iw.data[:n_freq, 0, 0].imag])
    freq_arr = np.asarray(freq_arr).T
    for num, (rfreqs, ifreqs) in enumerate(freq_arr):
        axes[1].plot(rfreqs, 's-.', label=str(num))
        axes[1].plot(ifreqs, 'o-.', label=str(num))
    graf = r'$G(i\omega_n)$'
    label_convergence(beta, u_str,
                      axes, graf, n_freq, xlim)


def list_show_conv(BETA, filestr='CH_sb_b{BETA}.h5', n_freq=5, xlim=2, skip=5):
    """Plots in individual figures for all interactions the DMFT loops"""
    with HDFArchive(filestr.format(BETA=BETA), 'r') as output_files:
        urange = output_files.keys()
    for u_str in urange:
        show_conv(BETA, u_str, filestr, n_freq, xlim, skip)
        plt.show()
        plt.close()


def phase_diag(beta_array, filestr='CH_sb_b{BETA}.h5'):

    for BETA in beta_array:
        tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
        with HDFArchive(filestr.format(BETA=BETA), 'r') as results:
            fl_dos = []
            for u_str in results.keys():
                lastit = results[u_str].keys()[-1]
                gf_iw = results[u_str][lastit]['giw']
                gf_iw = .5*(gf_iw['up']+gf_iw['down'])
                fl_dos.append(gf.fit_gf(w_n[:3], gf_iw.data[:, 0, 0].imag)(0.))

            u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
            temp = np.ones(len(u_range)) / BETA
            plt.scatter(u_range, temp, c=fl_dos,
                        s=150, vmin=-2, vmax=0, cmap=plt.get_cmap('inferno'))
    plt.title(r'Phase diagram single impurity')
    plt.xlabel('$U/D$')
    plt.ylabel(r'$T/D$')

def tail_clean(gf_iw, U):
    fixed = TailGf(1, 1, 3, 1)
    fixed[1] = np.array([[1]])
    fixed[3] = np.array([[U**2/4 + .25]])
    gf_iw.fit_tail(fixed, 5, int(gf_iw.beta), len(gf_iw.mesh))

def epot(BETA, filestr='CH_sb_b{BETA}.h5'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
    wsqr_4 = 4*w_n*w_n
    V = []
    with HDFArchive(filestr.format(BETA=BETA), 'r') as results:
        for u_str in results:
            lastit = results[u_str].keys()[-1]
            gf_iw = results[u_str][lastit]['giw']
            gf_iw = .5*(gf_iw['up']+gf_iw['down'])
            u_int = float(u_str[1:])
            tail_clean(gf_iw, u_int)
            gf_iw.data.real = 0.

            V.append(gf_iw.total_density()
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(V) - BETA*ur**2/32, ur
