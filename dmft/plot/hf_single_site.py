r"""
Plotting utilities for the Single site DMFT phase diagram
=========================================================

"""

from __future__ import division, print_function, absolute_import
import dmft.common as gf
import dmft.h5archive as h5
import dmft.hirschfye as hf
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22, 'figure.autolayout': True})


def label_convergence(beta, u_str, axes, graf, n_freq, xlim):
    """Label the axes of the common plot of the evolution of DMFT loops"""
    axes[0].set_xlim([0, xlim])
    axes[1].legend(loc=0, ncol=n_freq)
    axes[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, beta, u_str[1:]))
    axes[0].set_ylabel(graf)
    axes[0].set_xlabel(r'$i\omega_n$')
    axes[1].set_title('Evolution of the first frequencies')
    axes[1].set_ylabel(graf+'$(l)$')
    axes[1].set_xlabel('iterations')


def get_giw(h5parent, iteration, tau, w_n, tp):
    """Recovers with Fourier Transform G_iw from H5 file"""
    setup = h5.get_attributes(h5parent[iteration])
    mu, U = setup['MU'], setup['U']
    gtau = h5parent[iteration]['gtau'][:]
    giw = gf.gt_fouriertrans(gtau, tau, w_n,
                             [1., -mu, 0.25 + U**2/4])

    return gtau, giw


def show_conv(beta, u_str, filestr='SB_PM_B{}.h5', n_freq=5, xlim=2, skip=5):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    with h5.File(filestr.format(beta), 'r') as output_files:
        keys = output_files[u_str].keys()
        setup = h5.get_attributes(output_files[u_str][keys[-1]])
        tau, w_n = gf.tau_wn_setup(setup)
        for step in sorted(keys):
            gtau = output_files[u_str][step]['gtau'][:]
            giw = gf.gt_fouriertrans(gtau, tau, w_n)
            axes[0].plot(w_n, giw.imag)
            freq_arr.append(giw.imag[:n_freq])

    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        axes[1].plot(freqs, 'o-.', label=str(num))
    graf = r'$G(i\omega_n)$'

    label_convergence(beta, u_str, axes, graf, n_freq, xlim)

    return axes

def list_show_conv(beta, filestr='SB_PM_B{}.h5', n_freq=5, xlim=2, skip=5):
    """Plots in individual figures for all interactions the DMFT loops"""
    with h5.File(filestr.format(beta), 'r') as output_files:
        urange = output_files.keys()

    for u_str in urange:
        show_conv(beta, u_str, filestr, n_freq, xlim, skip)
        plt.show()
        plt.close()


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

    with h5.File(filestr.format(beta), 'r') as output_files:
        for u_str in sorted(output_files.keys()):
            u_range.append(float(u_str[1:]))
            last_iterations = sorted(output_files[u_str].keys())[-avg:]
            tau, w_n = gf.tau_wn_setup(h5.get_attributes(output_files[u_str][last_iterations[-1]]))
            gtau = hf.averager(output_files[u_str], 'gtau', last_iterations)
            giw = gf.gt_fouriertrans(gtau, tau, w_n, [1., 0., 0.25 + U**2/4])

            gfit = gf.fit_gf(w_n[:3], giw.imag)
            figiw.append(gfit)
            lgiw.append(giw)

    return np.asarray(u_range), figiw, lgiw


def plot_fit_dos(beta, avg, filestr='SB_PM_B{}.h5', xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8))

    u_range, fit_gf, lgiw = fit_dos(beta, avg, filestr)
    w_n = gf.matsubara_freq(beta, len(lgiw[0]))
    omega = np.arange(0, w_n[2], 0.05)
    for u_int, gfit, giw in zip(u_range, fit_gf, lgiw):
        axes[0].plot(w_n, giw.imag, 'o:', label='U='+str(u_int))
        axes[0].plot(omega, gfit(omega), 'k:')

    axes[0].set_xlim([0, xlim])
    axes[1].plot(u_range, [dos(0) for dos in fit_gf], 'o-')

    plt.show()
    plt.close()


def phases(beta_array):
    """Scatter plot of the DOS at Fermi level

    Shows the phase diagram of the impurity model of DMFT"""

    for beta in beta_array:
        u_int, gfit, _ = fit_dos(beta, 2)
        temp = np.ones(len(u_int)) * 2 / beta
        plt.scatter(u_int, temp, s=300, c=[dos(0) for dos in gfit],
                    vmin=-2, vmax=0)

    plt.xlabel('U/D')
    plt.ylabel('T/t')
