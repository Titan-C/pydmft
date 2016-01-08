# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse, TailGf
from pytriqs.gf.local import GfReFreq
from pytriqs.plot.mpl_interface import oplot
from pytriqs.archive import HDFArchive
from dmft.plot.hf_single_site import label_convergence
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, tp=0.25, filestr='DIMER_PM_B{BETA}_tp{tp}.h5',
              block=2, n_freq=2, xlim=2, skip=0, sig=False):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    with HDFArchive(filestr.format(tp=tp, BETA=beta), 'r') as datarecord:
        for step in datarecord[u_str].keys()[skip:]:
            labels = [name for name in datarecord[u_str][step]['G_iw'].indices]
            gf_iw = datarecord[u_str][step]['G_iw']
            u_int = float(u_str[1:])
            paramagnetic_hf_clean(gf_iw, u_int, tp)
            gf_iw = gf_iw[labels[block]]
            if sig:
                shift = 1. if 'asym' in labels[block] else -1
                gf_iw << iOmega_n + u_int/2. + shift * tp - 0.25 * gf_iw - inverse(gf_iw)

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
    label_convergence(beta, u_str+'\n$t_\\perp={}$'.format(tp),
                      axes, graf, n_freq, xlim)


def list_show_conv(BETA, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5',
                   block=2, n_freq=5, xlim=2, skip=5, sig=False):
    """Plots in individual figures for all interactions the DMFT loops"""
    with HDFArchive(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        urange = output_files.keys()

    for u_str in urange:
        show_conv(BETA, u_str, tp, filestr, block, n_freq, xlim, skip, sig)

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
                gf_iw = np.squeeze([gf_iw(i) for i in range(3)])
                fl_dos.append(gf.fit_gf(w_n[:3], gf_iw.imag)(0.))

            u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
            plt.scatter(np.ones(len(fl_dos))*tp, u_range, c=fl_dos,
                        s=150, vmin=-2, vmax=0, cmap=plt.get_cmap('inferno'))
    plt.xlim([0, 1])
    plt.title(r'Phase diagram at $\beta={}$'.format(BETA))
    plt.xlabel(r'$t_\perp/D$')
    plt.ylabel('$U/D$')

def averager(h5parent, h5child, last_iterations):
    """Given an H5 file parent averages over the iterations with the child"""
    sum_child = 0.
    for step in last_iterations:
        sum_child += h5parent[step][h5child]

    return 1. / len(last_iterations) * sum_child

def get_giw(h5parent, iteration_slice):
    """Recover G_iw from h5parent at iteration_slice

    Parameters
    ----------
    h5parent : h5py parent object
    iteration_slice : list or slice of iterations to average over

    Returns
    -------
    G_iw object
    """

    iterations = list(h5parent.keys())
    return averager(h5parent, 'G_iw', iterations[iteration_slice])


def tail_clean(gf_iw, U, tp):
    fixed = TailGf(1, 1, 3, 1)
    fixed[1] = np.array([[1]])
    fixed[2] = np.array([[-tp]])
    fixed[3] = np.array([[U**2/4 + tp**2 + .25]])
    gf_iw.fit_tail(fixed, 5, int(gf_iw.beta), len(gf_iw.mesh))


def paramagnetic_hf_clean(G_iw, u_int, tp):
    """Performs the average over up & dw of the green functions to
    enforce paramagnetism"""

    try:
        G_iw['asym_up'] << 0.5 * (G_iw['asym_up'] + G_iw['asym_dw'])
        tail_clean(G_iw['asym_up'], u_int, tp)

        G_iw['sym_up'] << 0.5 * (G_iw['sym_up'] + G_iw['sym_dw'])
        tail_clean(G_iw['sym_up'], u_int, -tp)

        G_iw['asym_dw'] << G_iw['asym_up']
        G_iw['sym_dw'] << G_iw['sym_up']

    except:
        G_iw['high_up'] << 0.5 * (G_iw['high_up'] + G_iw['high_dw'])
        tail_clean(G_iw['high_up'], u_int, tp)

        G_iw['low_up'] << 0.5 * (G_iw['low_up'] + G_iw['low_dw'])
        tail_clean(G_iw['low_up'], u_int, -tp)

        G_iw['high_dw'] << G_iw['high_up']
        G_iw['low_dw'] << G_iw['low_up']


def ekin(BETA, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):
    """Kinetic Energy per molecule"""
    T = []
    with HDFArchive(filestr.format(BETA=BETA, tp=tp), 'r') as results:
        for u_str in results:
            lastit = results[u_str].keys()[-3:]
            gf_iw = averager(results[u_str], 'G_iw', lastit)
            u_int = float(u_str[1:])
            paramagnetic_hf_clean(gf_iw, u_int, tp)

            gf_iw << 0.25*gf_iw*gf_iw
            T.append(gf_iw.total_density())
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(T), ur


def epot(BETA, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):
    """Potential energy per molecule"""
    V = []
    with HDFArchive(filestr.format(BETA=BETA, tp=tp), 'r') as results:
        for u_str in results:
            lastit = results[u_str].keys()[-3:]
            gf_iw = averager(results[u_str], 'G_iw', lastit)
            sig_iw= gf_iw.copy()
            u_int = float(u_str[1:])
            paramagnetic_hf_clean(gf_iw, u_int, tp)

            for name, g0block in gf_iw:
                shift = 1. if 'asym' or 'high' in name else -1
                sig_iw[name] << iOmega_n + u_int/2. + shift * tp - 0.25*gf_iw[name]- inverse(gf_iw[name])


            gf_iw << 0.5*sig_iw*gf_iw
            V.append(gf_iw.total_density())
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(V), ur
