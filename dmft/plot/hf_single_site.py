r"""
Plotting utilities for the Single site DMFT phase diagram
=========================================================

"""

from __future__ import division, print_function, absolute_import
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import dmft.common as gf
import dmft.ipt_imag as ipt
import dmft.h5archive as h5
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


def averager(sim_dir, observable, last_iterations):
    """Given an H5 file parent averages over the iterations with the child"""
    sum_child = 0.
    for step in last_iterations:
        sum_child += np.load(os.path.join(sim_dir, step, observable))

    return sum_child / len(last_iterations)


def gf_tail(gtau, U, mu):
    r"""Estimates the known first 3 moments of the tail

    Starting from the high energy expansion of [self-energy]_

    .. math:: \Sigma(i\omega_n \rightarrow \infty) = U \left(n_{\bar{\sigma}} - \frac{1}{2}\right)
        + \frac{U^2}{i\omega_n} n_{\bar{\sigma}} (1 - n_{\bar{\sigma}})

    Then in the Bethe lattice where the self consistency is
    :math:`-t^2G` the Green function decays as

    .. math:: G(i\omega_n \rightarrow \infty) = \frac{1}{i\omega_n}
        - \frac{\mu + U(\frac{1}{2} + G(\tau=0^+))}{(i\omega_n)^2}
        + \frac{t^2 + U^2/4 + \mu^2 - U\mu + 2U\mu G(\tau=0^+)}{(i\omega_n)^3}

    Parameters
    ----------
    gtau : Imaginary Time Green function array
    U : float Local interaction
    mu : float chemical potential

    Returns
    -------
    list

    References
    ----------

    .. [self-energy] : Gull, E. et al. Reviews of Modern Physics, 83(2), 384 http://dx.doi.org/10.1103/RevModPhys.83.349

    """
    g_t0 = gtau[0] if len(gtau.shape) == 1 else gtau[:, 0].reshape(2, 1)

    return [1., -mu - U*(0.5+g_t0), 0.25 + U**2/4 + mu**2 - U*mu + 2*U*mu*g_t0]


def interpol(gtau, Lrang, add_edge=False, same_particle=False):
    """This function interpolates :math:`G(\\tau)` onto a different array

    it keep track of the shape of the Greens functions in Beta^-.

    Parameters
    ----------
    gtau : ndarray
        Green function to interpolate
    Lrang : int
        number of points to describe
    add_edge : bool
        if the point Beta^- is missing add it
    same_particle : bool
        because fermion commutation relations if same fermion the
        edge has an extra -1
    """
    rtau = np.linspace(0, 1, gtau.size)
    if add_edge:
        gtau = np.concatenate((gtau, [-gtau[0]]))
        rtau = np.linspace(0, 1, gtau.size)  # update size
        if same_particle:
            gtau[-1] -= 1.
    interp = interp1d(rtau, gtau)
    nrang = np.linspace(0, 1, Lrang)
    return interp(nrang)


def get_giw(sim_dir, iteration_slice, tau, w_n):
    r"""Recovers with Fourier Transform G_iw from H5 file

    Parameters
    ----------
    h5parent : hdf5 group to go
    iteration_slice : list of iteration names to average over
    tau : 1D real array time slices of HF data
    w_n : 1D real array matsubara frequencies

    Returns
    -------
    tuple : :math:`G(\tau)`, :math:`G(i\omega_n)`
    """

    with open(sim_dir + '/setup', 'r') as read:
        setup = json.load(read)
    gtau = averager(sim_dir, 'gtau.npy', iteration_slice)
    giw = gf.gt_fouriertrans(gtau, tau, w_n,
                             gf_tail(gtau, setup['U'], setup['MU']))

    return gtau, giw


def get_sigmaiw(h5parent, iteration, tau, w_n):
    """Returns the self-energy with the Dyson equation"""
    _, giw = get_giw(h5parent, iteration, tau, w_n)
    return 1j*w_n - .25*giw - 1/giw


def show_conv(beta, u_str, filestr='SB_PM_B{}', n_freq=5, xlim=2, skip=5):
    """Plot the evolution of the Green's function in DMFT iterations"""
    freq_arr = []
    sim_dir = os.path.join(filestr.format(beta), u_str)
    iterations = sorted([it for it in os.listdir(sim_dir) if 'it' in it])[skip:]
    with open(sim_dir + '/setup', 'r') as read:
        setup = json.load(read)
    tau, w_n = gf.tau_wn_setup(setup)

    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    for step in iterations:
        _, giw = get_giw(sim_dir, [step], tau, w_n)
        if len(giw.shape) > 1:
            axes[0].plot(w_n, giw[0].real, 'gs:', w_n, giw[0].imag, 'bo:')
            freq_arr.append(np.array([giw[0].real[:n_freq], giw[0].imag[:n_freq]]))
        else:
            axes[0].plot(w_n, giw.imag)
            freq_arr.append(giw.imag[:n_freq])

    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        axes[1].plot(freqs.T, 'o-.', label=str(num))
    graf = r'$G(i\omega_n)$'

    label_convergence(beta, u_str, axes, graf, n_freq, xlim)

    return axes


def list_show_conv(beta, filestr='SB_PM_B{}', n_freq=5, xlim=2, skip=5):
    """Plots in individual figures for all interactions the DMFT loops"""
    urange = sorted([u_str for u_str in os.listdir(filestr.format(beta))
                     if 'U' in u_str])

    for u_str in urange:
        show_conv(beta, u_str, filestr, n_freq, xlim, skip)
        plt.show()
        plt.close()


def fit_dos(beta, avg, filestr='SB_PM_B{}'):
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

    ulist = sorted([u_str for u_str in os.listdir(filestr.format(beta))
                     if 'U' in u_str])
    for u_str in ulist:
        sim_dir = os.path.join(filestr.format(beta), u_str)
        last_iterations = sorted([it for it in os.listdir(sim_dir) if 'it' in it])[-avg:]
        with open(sim_dir + '/setup', 'r') as read:
            setup = json.load(read)
            tau, w_n = gf.tau_wn_setup(setup)

        _, giw = get_giw(sim_dir, last_iterations, tau, w_n)

        gfit = gf.fit_gf(w_n[:3], giw.imag)
        figiw.append(gfit)
        lgiw.append(giw)

    u_range = np.asarray([float(u_str[1:]) for u_str in ulist])
    return u_range, figiw, lgiw


def plot_fit_dos(beta, avg, filestr='SB_PM_B{}', xlim=2):
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
