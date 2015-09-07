r"""
Single site review
==================

Searching for the coexistence region using the Hirsch-Fye impurity solver
for the single site DMFT problem

"""


from __future__ import division, print_function, absolute_import
from pytriqs.archive import HDFArchive
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
import re
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, filestr='SB_PM_B{}.h5', nf=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    f, ax = plt.subplots(1, 2, figsize=(13, 8))
    freq_arr = []
    with HDFArchive(filestr.format(beta), 'r') as output_files:
        w_n = gf.matsubara_freq(output_files[u_str]['it00']['setup']['BETA'],
                                output_files[u_str]['it00']['setup']['N_MATSUBARA'])
        for it in sorted(output_files[u_str].keys()):
            giw = output_files[u_str][it]['giw']
            ax[0].plot(w_n, giw.imag)
            freq_arr.append(giw.imag[:nf])

    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        ax[1].plot(freqs, 'o-.', label=str(num))
    ax[0].set_xlim([0, 2])
    ax[1].legend(loc=0, ncol=nf)
    graf = r'$G(i\omega_n)$'
    ax[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, beta, u_str[1:]))
    ax[0].set_ylabel(graf)
    ax[0].set_xlabel(r'$i\omega_n$')
    ax[1].set_title('Evolution of the first frequencies')
    ax[1].set_ylabel(graf+'$(l)$')
    ax[1].set_xlabel('iterations')

    plt.show()
    plt.close()


def list_show_conv(beta, filestr='SB_PM_B{}.h5', nf=5, xlim=2):
    with HDFArchive(filestr.format(beta), 'r') as output_files:
        urange = output_files.keys()

    for u_str in urange:
        show_conv(beta, u_str, filestr, nf, xlim)

def _averager(it_output, last_iterations):
    """Averages over the files terminating with the numbers given in vector"""
    sgiw = 0
    for it in last_iterations:
        giw = it_output[it]['giw']
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

    U = []
    figiw = []
    lgiw = []
    w_n = gf.matsubara_freq(beta, 3)

    with HDFArchive(filestr.format(beta), 'r') as output_files:
        for u_str in sorted(output_files.keys()):
            U.append(float(u_str[1:]))
            last_iterations = sorted(output_files[u_str].keys())[-avg:]
            giw = _averager(output_files[u_str], last_iterations)

            gfit = gf.fit_gf(w_n, giw.imag)
            figiw.append(gfit)
            lgiw.append(giw)

    return np.asarray(U), figiw, lgiw


def plot_fit_dos(beta, avg, filestr='SB_PM_B{}.h5', xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    f, ax = plt.subplots(1, 2, figsize=(13, 8))

    U, fit_gf, lgiw = fit_dos(beta, avg)
    wn = gf.matsubara_freq(beta, 512)
    wr = np.arange(0, wn[2], 0.05)
    for u, gfit, giw in zip(U, fit_gf, lgiw):
        ax[0].plot(wn, giw.imag, 'o:', label='U='+str(u))
        ax[0].plot(wr, gfit(wr), 'k:')

    ax[0].set_xlim([0, xlim])
    ax[1].plot(U, [dos(0) for dos in fit_gf], 'o-')

    plt.show()
    plt.close()


def phases():
    fig, axs = plt.subplots()
    for beta in np.array([32., 40., 64.]):
        U, gfit, _ = fit_dos(beta, 2)
        T = np.ones(len(U)) * 2 / beta
        axs.scatter(U, T, s=300, c=[dos(0) for dos in gfit])

    plt.xlabel('U/D')
    plt.ylabel('T/t')
    print(U)
