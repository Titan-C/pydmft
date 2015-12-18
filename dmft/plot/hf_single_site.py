r"""
Plotting utilities for the Single site DMFT phase diagram
=========================================================

"""

from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import dmft.common as gf
import dmft.ipt_imag as ipt
import dmft.h5archive as h5
import dmft.hirschfye as hf
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


def get_giw(h5parent, iteration, tau, w_n):
    """Recovers with Fourier Transform G_iw from H5 file"""
    setup = h5.get_attributes(h5parent[iteration])
    mu, U = setup['MU'], setup['U']
    gtau = h5parent[iteration]['gtau'][:]
    giw = gf.gt_fouriertrans(gtau, tau, w_n,
                             [1., -mu, 0.25 + U**2/4])

    return gtau, giw

def get_sigmaiw(h5parent, iteration, tau, w_n):
    """Returns the self-energy with the Dyson equation"""
    _, giw = get_giw(h5parent, iteration, tau, w_n)
    return 1j*w_n - .25*giw - 1/giw


def show_conv(beta, u_str, filestr='SB_PM_B{}.h5', n_freq=5, xlim=2, skip=5):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    with h5.File(filestr.format(beta), 'r') as output_files:
        keys = output_files[u_str].keys()[skip:]
        setup = h5.get_attributes(output_files[u_str][keys[-1]])
        tau, w_n = gf.tau_wn_setup(setup)
        for step in keys:
            _, giw = get_giw(output_files[u_str], step, tau, w_n)
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
            U = float(u_str[1:])
            u_range.append(U)
            last_iterations = sorted(output_files[u_str].keys())[-avg:]
            tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=beta))
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
        temp = np.ones(len(u_int)) / beta
        plt.scatter(u_int, temp, s=300, c=[dos(0) for dos in gfit],
                    vmin=-2, vmax=0)

    plt.xlabel('U/D')
    plt.ylabel('T/t')


def energies(beta, filestr='SB_PM_B{}.h5'):
    """returns the potential, and kinetic energy

    Parameters
    ----------
    beta : float, inverse temperature
    filestr : string, results file name, beta is replaced in format

    Returns
    -------
    tuple of 3 ndarrays
        Contains, potential energy, Kinetic energy, values of U
    """
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=beta))
    giw_free = gf.greenF(w_n)
    e_mean = ipt.e_mean(beta)
    ekin = []
    epot = []
    with h5.File(filestr.format(beta), 'r') as results:
        for u_str in results:
            last_iter = results[u_str].keys()[-1]
            _, giw = get_giw(results[u_str], last_iter, tau, w_n)
            siw = get_sigmaiw(results[u_str], last_iter, tau, w_n)

            u_int = float(u_str[1:])

            epot.append(ipt.epot(giw, siw, u_int, beta, w_n))
            ekin.append(ipt.ekin(giw, siw, beta, w_n, e_mean, giw_free))
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(epot), np.array(ekin), ur
