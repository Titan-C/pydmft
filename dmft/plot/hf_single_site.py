r"""
Plotting utilities for the Single site DMFT phase diagram
=========================================================

"""

from __future__ import division, print_function, absolute_import
from pytriqs.archive import HDFArchive
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, filestr='SB_PM_B{}.h5', n_freq=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8))
    freq_arr = []
    with HDFArchive(filestr.format(beta), 'r') as output_files:
        w_n = gf.matsubara_freq(output_files[u_str]['it00']['setup']['BETA'],
                                output_files[u_str]['it00']['setup']['N_MATSUBARA'])
        for step in sorted(output_files[u_str].keys()):
            giw = output_files[u_str][step]['giw']
            axes[0].plot(w_n, giw.imag)
            freq_arr.append(giw.imag[:n_freq])

    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        axes[1].plot(freqs, 'o-.', label=str(num))
    axes[0].set_xlim([0, xlim])
    axes[1].legend(loc=0, ncol=n_freq)
    graf = r'$G(i\omega_n)$'
    axes[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, beta, u_str[1:]))
    axes[0].set_ylabel(graf)
    axes[0].set_xlabel(r'$i\omega_n$')
    axes[1].set_title('Evolution of the first frequencies')
    axes[1].set_ylabel(graf+'$(l)$')
    axes[1].set_xlabel('iterations')

    plt.show()
    plt.close()


def list_show_conv(beta, filestr='SB_PM_B{}.h5', n_freq=5, xlim=2):
    """Provides a list of all convergence plots at a given temperature"""
    with HDFArchive(filestr.format(beta), 'r') as output_files:
        urange = output_files.keys()

    for u_str in urange:
        show_conv(beta, u_str, filestr, n_freq, xlim)


def _averager(it_output, last_iterations):
    """Averages over the files terminating with the numbers given in vector"""
    sgiw = 0
    for step in last_iterations:
        giw = it_output[step]['giw']
        sgiw += giw

    sgiw /= len(last_iterations)
    sgiw.real = 0.
    return sgiw


def fit_dos(beta, avg, filestr='SB_PM_B{}.h5'):
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

    u_range = []
    figiw = []
    lgiw = []
    w_n = gf.matsubara_freq(beta, 3)

    with HDFArchive(filestr.format(beta), 'r') as output_files:
        for u_str in sorted(output_files.keys()):
            u_range.append(float(u_str[1:]))
            last_iterations = sorted(output_files[u_str].keys())[-avg:]
            giw = _averager(output_files[u_str], last_iterations)

            gfit = gf.fit_gf(w_n, giw.imag)
            figiw.append(gfit)
            lgiw.append(giw)

    return np.asarray(u_range), figiw, lgiw


def plot_fit_dos(beta, avg, filestr='SB_PM_B{}.h5', xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8))

    u_range, fit_gf, lgiw = fit_dos(beta, avg, filestr)
    w_n = gf.matsubara_freq(beta, 512)
    omega = np.arange(0, w_n[2], 0.05)
    for u_int, gfit, giw in zip(u_range, fit_gf, lgiw):
        axes[0].plot(w_n, giw.imag, 'o:', label='U='+str(u_int))
        axes[0].plot(omega, gfit(omega), 'k:')

    axes[0].set_xlim([0, xlim])
    axes[1].plot(u_range, [dos(0) for dos in fit_gf], 'o-')

    plt.show()
    plt.close()


def phases():
    """Scatter plot of the DOS at Fermi level

    Shows the phase diagram of the impurity model of DMFT"""

    for beta in np.array([32., 40., 50., 64.]):
        u_int, gfit, _ = fit_dos(beta, 2)
        temp = np.ones(len(u_int)) * 2 / beta
        plt.scatter(u_int, temp, s=300, c=[dos(0) for dos in gfit],
                    vmin=-2, vmax=0)

    plt.xlabel('U/D')
    plt.ylabel('T/t')
