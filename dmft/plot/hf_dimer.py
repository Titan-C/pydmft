# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:08:23 2015

@author: oscar
"""

from __future__ import division, print_function, absolute_import
from dmft.plot.hf_single_site import label_convergence
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.h5archive as h5
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from random import choice
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22, 'figure.autolayout': True})


def get_giw(h5parent, iteration, tau, w_n):
    """Recovers with Fourier Transform G_iw from H5 file

    Parameters
    ----------
    h5parent : h5py parent object
    iteration : string
        group where the measurements are stored, name of iteration
    tau : real float array
            Imaginary time points
    w_n : real float array
            fermionic matsubara frequencies. Only use the positive ones

    Returns
    -------
    tuple complex ndarray
            Interacting Greens function in matsubara frequencies
            diagonal and off diagonal


    See also
    --------
    get_sigmaiw
    """

    setup = h5.get_attributes(h5parent[iteration])
    mu, tp, U = setup['MU'], setup['tp'], setup['U']
    gtau_d = h5parent[iteration]['gtau_d'][:]
    gtau_o = h5parent[iteration]['gtau_o'][:]
    giw_d = gf.gt_fouriertrans(gtau_d, tau, w_n,
                               [1., -mu, 0.25 + U**2/4 + tp**2 + mu**2])
    giw_o = gf.gt_fouriertrans(gtau_o, tau, w_n, [0., tp, -10.*mu*tp**2])

    return giw_d, giw_o


def get_sigmaiw(h5parent, iteration, tau, w_n):
    """Calculates the Self-Energy in Matsubara Frequencies by the
    Dyson equation from the Monte Carlo data

    Parameters
    ----------
    h5parent : h5py parent object
    iteration : string
        group where the measurements are stored, name of iteration
    tau : real float array
            Imaginary time points
    w_n : real float array
            fermionic matsubara frequencies. Only use the positive ones

    Returns
    -------
    tuple complex ndarray
            Self-Energy in matsubara frequencies
            diagonal and off diagonal

    See also
    --------
    get_giw
    """

    giw_d, giw_o = get_giw(h5parent, iteration, tau, w_n)
    giw_d_inv, giw_o_inv = rt.mat_inv(giw_d, giw_o)

    setup = h5.get_attributes(h5parent[iteration])
    tp, t = setup['tp'], setup['t']

    sigmaiw_d = 1j*w_n - t**2 * giw_d - giw_d_inv
    sigmaiw_o = -tp    - t**2 * giw_o - giw_o_inv

    return sigmaiw_d, sigmaiw_o


def show_conv(BETA, u_str, tp=0.25, filestr='tp{tp}_B{BETA}.h5',
              n_freq=5, xlim=2, skip=5):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arrd = []
    freq_arro = []
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        setup = h5.get_attributes(output_files[u_str]['it000'])
        tau, w_n = gf.tau_wn_setup(setup)
        for step in output_files[u_str].keys()[skip:]:
            giwd, giwo = get_giw(output_files[u_str], step, tau, w_n)
            axes[0].plot(w_n, giwd.imag, 'bo:')
            axes[0].plot(w_n, giwo.real, 'gs:')

            freq_arrd.append(giwd.imag[:n_freq])
            freq_arro.append(giwo.real[:n_freq])

    freq_arrd = np.asarray(freq_arrd).T
    freq_arro = np.asarray(freq_arro).T

    for num, (freqsd, freqso) in enumerate(zip(freq_arrd, freq_arro)):
        axes[1].plot(freqsd, 'o-.', label=str(num))
        axes[1].plot(freqso, 's-.', label=str(num))

    labimgiws = mlines.Line2D([], [], color='blue', marker='o',
                              label=r'$\Im m G_{AA}$')
    labregiws = mlines.Line2D([], [], color='green', marker='s',
                              label=r'$\Re e G_{AB}$')
    axes[0].legend(handles=[labimgiws, labregiws], loc=0)

    graf = r'$G(i\omega_n)$'
    label_convergence(BETA, u_str+'\n$t_\\perp={}$'.format(tp),
                      axes, graf, n_freq, xlim)

    return axes

def list_show_conv(BETA, tp, filestr='tp{}_B{}.h5', n_freq=5, xlim=2, skip=5):
    """Plots in individual figures for all interactions the DMFT loops"""
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        urange = output_files.keys()

    for u_str in urange:
        show_conv(BETA, u_str, tp, filestr, n_freq, xlim, skip)
        docc, acc = report_docc_acc(BETA, u_str, tp, filestr)

        plt.show()
        plt.close()
        try:
            print('Last step double occupation: {:.6}'.format(docc),
              'The acceptance rate is:{:.1%}'.format(acc))
        except ValueError:
            pass


def plot_it(BETA, u_str, tp, it, space, label='', filestr='SB_PM_B{}.h5', axes=None):
    """Plot the evolution of the Green's function in DMFT iterations

    Parameters
    ----------
    it: int, iteration to plot
    space: string, tau or iw
    label: string to identify plot
    axes: Matplotlib axes, use to superpose plots

    Returns
    -------
    Matplotlig axes
    """


    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, sharey=True)
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        step = 'it{:03}'.format(it)
        setup = h5.get_attributes(output_files[u_str][step])
        tau, w_n = gf.tau_wn_setup(setup)

        gtau_d = output_files[u_str][step]['gtau_d'][:]
        gtau_o = output_files[u_str][step]['gtau_o'][:]
        if label == '':
            label = step

        if space == 'tau':
            axes[0].plot(tau, gtau_d.T, '-', label=label)
            axes[1].plot(tau, gtau_o.T, '-', label=label)

            graf = r'$G(\tau)$'
            axes[0].set_xlabel(r'$\tau$')
            axes[0].set_xlim([0, BETA])
        else:
            giw_d, giw_o = get_giw(output_files[u_str], step, tau, w_n)
            #c = choice(colors)
            #axes[0].plot(w_n, giw_d.real.T, c + 'o-', label='R ' + label)
            axes[0].plot(w_n, giw_d.imag.T, 's:', label='I ' + label)
            axes[1].plot(w_n, giw_o.real.T, 'o-', label='R ' + label)
            #axes[1].plot(w_n, giw_o.imag.T, c + 's:', label='I ' + label)

            graf = r'$G(i\omega_n)$'
            axes[0].set_xlabel(r'$i\omega$')

    axes[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, BETA, u_str[1:]))
    axes[0].set_ylabel(graf+'$_{AA}$')
    axes[0].legend(loc=0)
    axes[1].set_ylabel(graf+'$_{AB}$')
    axes[1].legend(loc=0)

    return axes


def report_docc_acc(BETA, u_str, tp, filestr):
    """Returns the last iteration mean double occupation and acceptance rate"""

    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        last_iter = output_files[u_str].keys()[-1]
        try:
            docc = output_files[u_str][last_iter]['double_occ'][:].mean()
            acc = output_files[u_str][last_iter]['acceptance'].value
        except KeyError:
            docc, acc = -1., -1.

    return docc, acc

def ekin(BETA, tp=0.25, filestr='tp{tp}_B{BETA}.h5',):
    e_mean = rt.free_ekin(tp, BETA)
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
    giw_free_d, _ = rt.gf_met(w_n, 0., tp, 0.5, 0.)
    T = []

    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as results:
        for u_str in results:
            last_iter = results[u_str].keys()[-1]
            giwd, giwo = get_giw(results[u_str], last_iter, tau, w_n)
            siwd, siwo = get_sigmaiw(results[u_str], last_iter, tau, w_n)

            T.append(2*(w_n*(giw_free_d - giwd).imag +
                     giwd.imag*siwd.imag - giwo.real*siwo.real).sum()/BETA + e_mean)
        ur = np.array([float(u_str[1:]) for u_str in results])
    return np.array(T), ur


def epot(BETA, tp=0.25, filestr='tp{tp}_B{BETA}.h5',):
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
    wsqr_4 = 4*w_n*w_n
    V = []
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as results:
        for u_str in results:
            last_iter = results[u_str].keys()[-1]
            giwd, giwo = get_giw(results[u_str], last_iter, tau, w_n)
            siwd, siwo = get_sigmaiw(results[u_str], last_iter, tau, w_n)

            u_int = float(u_str[1:])

            V.append((giwo*siwo + giwd*siwd + u_int**2/wsqr_4).real.sum()/BETA)
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(V) - BETA*ur**2/32, ur

def docc_plot(BETA, tp, filestr, ax=None):
    """Plots double occupation"""
    if ax is None:
        _, ax = plt.subplots()
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        docc = []
        for u_str in output_files:
            last_iter = output_files[u_str].keys()[-1]
            try:
                docc.append([float(u_str[1:]),
                             output_files[u_str][last_iter]['double_occ'][:].mean()])
            except KeyError:
                docc = np.nan

        docc=np.array(docc).T
        ax.scatter(docc[0], docc[1], c=docc[1],
                    s=120, marker='<', vmin=0, vmax=0.2)
        ax.plot(docc[0], docc[1], ':')
    ax.set_title(r'double occupation @'
                 r'$\beta={}$, tp={}'.format(BETA, tp))
    ax.set_ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
    ax.set_xlabel('U/D')
    return ax


def dos_plot(BETA, tp, filestr, ax=None):
    """Plots double occupation"""
    if ax is None:
        _, ax = plt.subplots()
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as results:
        fl_dos = []
        tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
        for u_str in results:
            lastit = results[u_str].keys()[-1]
            giwd, _ = get_giw(results[u_str], lastit, tau, w_n)
            fl_dos.append(-1./np.pi*gf.fit_gf(w_n[:3], giwd.imag)(0.))

        u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
        ax.scatter(u_range, fl_dos,
                    s=120, marker='>', vmin=0, vmax=2./np.pi)
    ax.set_title('Hysteresis loop of the \n density of states')
    ax.set_ylabel(r'$A(\omega=0)$')
    ax.set_xlabel('U/D')


def plot_acc(BETA, u_str, tp, filestr, skip=5):
    """Plot the evolution of the acceptance rate in each DMFT loop"""
    acceptance_log = []
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        for it_name in output_files[u_str].keys()[skip:]:
            try:
                acceptance_log.append(output_files[u_str][it_name]['acceptance'].value)
            except KeyError:
                acceptance_log.append(0.)

    plt.plot(acceptance_log, 'o-')
    plt.title(r'Change of acceptance @ $\beta={}$, U={}'.format(BETA, u_str[1:]))
    plt.ylabel('Acceptance rate')
    plt.xlabel('iterations')


def plot_tails(BETA, U, tp, ax=None):
    w_n = gf.matsubara_freq(BETA, BETA, BETA/2.)
    if ax is None:
        ax = plt
    ax.plot(w_n, -1/w_n, '--')
    ax.plot(w_n, -tp/w_n**2, '--')
    ax.plot(w_n, -1/w_n + U**2/4/w_n**3, '--')


def phase_diag_b(BETA_range, tp, filestr='HF_DIM_tp{tp}_B{BETA}.h5'):

    for BETA in BETA_range:
        tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
        with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as results:
            fl_dos = []
            for u_str in results.keys():
                lastit = results[u_str].keys()[-1]
                giwd, _ = get_giw(results[u_str], lastit, tau, w_n)
                fl_dos.append(gf.fit_gf(w_n[:3], giwd.imag)(0.))

            u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
            plt.scatter(u_range, np.ones(len(fl_dos))/BETA, c=fl_dos,
                        s=150, vmin=-2, vmax=0)
    plt.ylim([0, 0.04])
    plt.title(r'Phase diagram at $t_\perp={}$'.format(tp))
    plt.ylabel(r'$T/D$')
    plt.xlabel('$U/D$')


def phase_diag(BETA, tp_range, filestr='HF_DIM_tp{tp}_B{BETA}.h5'):

    for tp in tp_range:
        tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
        with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as results:
            fl_dos = []
            for u_str in results.keys():
                lastit = results[u_str].keys()[-1]
                giwd, _ = get_giw(results[u_str], lastit, tau, w_n)
                fl_dos.append(gf.fit_gf(w_n[:3], giwd.imag)(0.))

            u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
            plt.scatter(np.ones(len(fl_dos))*tp, u_range, c=fl_dos,
                        s=150, vmin=-2, vmax=0, cmap=plt.get_cmap('inferno'))
    plt.xlim([0, 1])
    plt.title(r'Phase diagram at $\beta={}$'.format(BETA))
    plt.xlabel(r'$t_\perp/D$')
    plt.ylabel('$U/D$')


def spectral(tp, U, BETA, pade_fit_pts):
    rot = np.matrix([[-1, 1], [1, 1]])/np.sqrt(2)
    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, BETA)
    f, (gl, gd) = plt.subplots(1, 2, figsize=(18, 8))
    with h5.File(filestr, 'r') as results:
        u = 'U'+str(U)
        lastit = results[u].keys()[-1]
        g_iw = getGiw(results[u][lastit])
        greal = GfReFreq(indices=[0, 1], window=(-3.5, 3.5), n_points=500)
        greal.set_from_pade(g_iw, pade_fit_pts, 0.)
        gl.oplot(greal[0, 0], RI='S', label='out')
        gl.set_title('On site GF, fit pts'+str(pade_fit_pts))
        gl.set_ylim([0, 0.6])

        rgiw = rot*g_iw*rot
        greal.set_from_pade(rgiw, pade_fit_pts, 0.)
        gd.oplot(greal[0, 0], RI='S', label='bond')
        gd.oplot(greal[1, 1], RI='S', label='anti-bond')
        gd.set_title('Diagonal GF')

        gl.oplot(0.5*(greal[0, 0] + greal[1, 1]), '--', RI='S', label='d avg')

    plt.show()
    plt.close()
