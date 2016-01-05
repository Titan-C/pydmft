# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse, GfReFreq, \
    TailGf, SemiCircular, GfImTime, InverseFourier
from pytriqs.plot.mpl_interface import oplot
import dmft.common as gf
import dmft.plot.cthyb_h_single_site as pss
from dmft.plot.hf_single_site import label_convergence

plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, filestr='CH_sb_b{BETA}.h5', n_freq=2, xlim=2, skip=0, sig=False):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    with HDFArchive(filestr.format(BETA=beta), 'r') as datarecord:
        for step in datarecord[u_str].keys()[skip:]:
            gf_iw = datarecord[u_str][step]['giw']
            gf_iw = .5*(gf_iw['up']+gf_iw['down'])
            if sig:
                u_int = float(u_str[1:])
                gf_iw.data.real = 0.
                tail_clean(gf_iw, u_int)
                gf_iw << iOmega_n + u_int/2. - 0.25 * gf_iw - inverse(gf_iw)
            axes[0].oplot(gf_iw.imag, 'bo:', label=None)
            axes[0].oplot(gf_iw.real, 'gs:', label=None)
            gf_iw = np.squeeze([gf_iw(i) for i in range(n_freq)])
            freq_arr.append([gf_iw.real, gf_iw.imag])
    freq_arr = np.asarray(freq_arr).T
    for num, (rfreqs, ifreqs) in enumerate(freq_arr):
        axes[1].plot(rfreqs, 's-.', label=str(num))
        axes[1].plot(ifreqs, 'o-.', label=str(num))
    graf = r'$G$' if not sig else r'$\Sigma$'
    graf += r'$(i\omega_n)$'
    label_convergence(beta, u_str,
                      axes, graf, n_freq, xlim)
    return axes


def list_show_conv(BETA, filestr='CH_sb_b{BETA}.h5', n_freq=5, xlim=2, skip=5, sig=False):
    """Plots in individual figures for all interactions the DMFT loops"""
    with HDFArchive(filestr.format(BETA=BETA), 'r') as output_files:
        urange = output_files.keys()
    for u_str in urange:
        show_conv(BETA, u_str, filestr, n_freq, xlim, skip, sig)
        plt.show()
        plt.close()


def phase_diag(beta_array, filestr='CH_sb_b{BETA}.h5'):

    for BETA in beta_array:
        w_n = gf.matsubara_freq(BETA, 3)
        with HDFArchive(filestr.format(BETA=BETA), 'r') as results:
            fl_dos = []
            for u_str in results.keys():
                lastit = results[u_str].keys()[-1]
                gf_iw = results[u_str][lastit]['giw']
                gf_iw = .5*(gf_iw['up']+gf_iw['down'])
                gf_iw = np.squeeze([gf_iw(i) for i in range(3)])
                fl_dos.append(gf.fit_gf(w_n, gf_iw.imag)(0.))

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
    V = []
    with HDFArchive(filestr.format(BETA=BETA), 'r') as results:
        for u_str in results:
            lastit = results[u_str].keys()[-1]
            gf_iw = results[u_str][lastit]['giw']
            gf_iw = .5*(gf_iw['up']+gf_iw['down'])
            u_int = float(u_str[1:])
            gf_iw.data.real = 0.
            tail_clean(gf_iw, u_int)

            sig_iw = iOmega_n + u_int/2. - 0.25 * gf_iw - inverse(gf_iw)

            gf_iw << gf_iw*sig_iw
            V.append(0.5*(gf_iw).total_density())
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(V), ur


def ekin(BETA, filestr='CH_sb_b{BETA}.h5'):
    T = []
    with HDFArchive(filestr.format(BETA=BETA), 'r') as results:
        for u_str in results:
            lastit = results[u_str].keys()[-1]
            gf_iw = results[u_str][lastit]['giw']
            gf_iw = .5*(gf_iw['up']+gf_iw['down'])
            gf_iw.data.real = 0.
            u_int = float(u_str[1:])
            tail_clean(gf_iw, u_int)
            gf_iw << .25*gf_iw*gf_iw

            T.append(gf_iw.total_density())

        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.asarray(T), ur
