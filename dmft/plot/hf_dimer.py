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
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22, 'figure.autolayout': True})


def get_giw(h5parent, iteration, tau, w_n, tp):
    """Recovers with Fourier Transform G_iw from H5 file"""
    gtau_d = h5parent[iteration]['gtau_d'][:]
    gtau_o = h5parent[iteration]['gtau_o'][:]
    giw_d = gf.gt_fouriertrans(gtau_d, tau, w_n)
    giw_o = gf.gt_fouriertrans(gtau_o, tau, w_n, [0., tp, 0.])

    return giw_d, giw_o


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
            giwd, giwo = get_giw(output_files[u_str], step,
                                 tau, w_n, setup['tp'])
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
        print('Last step double occupation: {:.6}'.format(docc),
              'The acceptance rate is:{:.1%}'.format(acc))


def plot_it(BETA, u_str, tp, skip, it, space, label='', filestr='SB_PM_B{}.h5', axes=None):
    """Plot the evolution of the Green's function in DMFT iterations

    Parameters
    ----------
    skip: int, -1 is last iteration
    space: string, tau or iw
    label: string to identify plot
    axes: Matplotlib axes, use to superpose plots

    Returns
    -------
    Matplotlig axes
    """

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, sharey=True)
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        setup = h5.get_attributes(output_files[u_str]['it001'])
        tau, w_n = gf.tau_wn_setup(setup)
        for step in output_files[u_str].keys()[skip:it]:
            gtau_d = output_files[u_str][step]['gtau_d'][:]
            gtau_o = output_files[u_str][step]['gtau_o'][:]
            if space == 'tau':
                axes[0].plot(tau, gtau_d, '-', label=label)
                axes[1].plot(tau, gtau_o, '-', label=label)
            else:
                giw_d = gf.gt_fouriertrans(gtau_d, tau, w_n)
                giw_o = gf.gt_fouriertrans(gtau_o, tau, w_n, [0., tp, 0.])
                axes[0].plot(w_n, giw_d.imag, 'o:', label=label)
                axes[1].plot(w_n, giw_o.real, 's:', label=label)

        if space == 'tau':
            graf = r'$G(\tau)$'
            axes[0].set_xlabel(r'$\tau$')
            axes[0].set_xlim([0, BETA])
        else:
            graf = r'$G(i\omega_n)$'
            axes[0].set_xlabel(r'$i\omega$')

    axes[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, BETA, u_str[1:]))
    axes[0].set_ylabel(graf+'$_{AA}$')
    axes[1].set_ylabel(graf+'$_{AB}$')

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


def docc_plot(BETA, tp, filestr):
    """Plots double occupation"""
    with h5.File(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        for u_str in output_files:
            last_iter = output_files[u_str].keys()[-1]
            try:
                docc = output_files[u_str][last_iter]['double_occ'][:].mean()
                marker = 'bo' if docc>0.05 else 'rs'
                plt.plot(float(u_str[1:]), docc, marker)
            except KeyError:
                docc = np.nan

    plt.title('Hysteresis loop of the double occupation')
    plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
    plt.xlabel('U/D')


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


def get_selfE(gtau_d, gtau_o):
    nf = len(gtau_d.mesh)
    beta = gtau_d.beta
    g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=nf)
    rt.load_gf(g_iw, gtau_d, gtau_o)
    gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, 0, 0.2)
    sigma = g_iw.copy()
    sigma << gmix - 0.25*g_iw - inverse(g_iw)
    return sigma


def plot_tails(BETA, U, tp, ax=None):
    w_n = gf.matsubara_freq(BETA, BETA, BETA/2.)
    if ax is None:
        ax = plt
    ax.plot(w_n, -1/w_n, '--')
    ax.plot(w_n, -tp/w_n**2, '--')
    ax.plot(w_n, -1/w_n + U**2/4/w_n**3, '--')


def phase_diag(BETA):

    fl_dos = []
    w_n = gf.matsubara_freq(BETA, 3)
    for tp in np.arange(0.18, 0.3, 0.01):
        filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, BETA)
        with h5.File(filestr, 'r') as results:
            for u in results.keys():
                lastit = results[u].keys()[-1]
                fl_dos.append(gf.fit_gf(w_n, results[u][lastit]['gtau_d'].imag)(0.))
    return np.asarray(fl_dos)


def compare_last_gf(tp, BETA, contl=2, ylim=-1):
    """Compartes in a plot the last GF of the HF simulation with the one CTHYB
       run done with this seed"""

    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, BETA)
    w_n = gf.matsubara_freq(BETA, 3)
    f, (gd, go) = plt.subplots(1, 2, figsize=(18, 8))
    with h5.File(filestr, 'r') as results:
        for u in results.keys():
            lastit = results[u].keys()[-1]
            gd.oplot(results[u][lastit]['gtau_d'], 'x-', RI='I', label=u)
            go.oplot(results[u][lastit]['gtau_o'], '+-', RI='R', label=u)
            giw = results[u][lastit]['gtau_d']

            fit = gf.fit_gf(w_n, giw[0, 0].imag)
            w = np.arange(0, contl, 0.05)
            gcont = fit(w)
            gd.plot(w, gcont, 'k:')

    gd.set_xlim([0, 4])
    gd.set_ylim([ylim, 0])
    gd.legend(loc=0, prop={'size': 18})
    gd.set_ylabel(r'$\Im m G_{AA}(i\omega_n)$')
    go.set_xlim([0, 4])
    go.legend(loc=0, prop={'size': 18})
    go.set_ylabel(r'$\Re e G_{AB}(i\omega_n)$')
    plt.suptitle('Matsubara GF $t_{{ab}}/D={}$ $\\beta D={}$'.format(tp, BETA))


def diag_sys(tp, BETA):
    rot = np.matrix([[-1, 1], [1, 1]])/np.sqrt(2)
    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, BETA)
    f, (gd, go) = plt.subplots(1, 2, figsize=(18, 8))
    with rt.h5.File(filestr, 'r') as results:
        for u in results.keys():
            lastit = results[u].keys()[-1]
            g_iw = rot*getGiw(results[u][lastit])*rot
            gd.oplot(g_iw['A', 'A'], 'x-', label=u)
            go.oplot(g_iw['B', 'B'], '+-', label=u)

    gd.set_xlim([0, 4])
    gd.legend(loc=0, prop={'size': 18})
    gd.set_ylabel(r'$\Im m G_{1}(i\omega_n)$')
    go.set_xlim([0, 4])
    go.legend(loc=0, prop={'size': 18})
    go.set_ylabel(r'$\Re e G_{2}(i\omega_n)$')
    plt.suptitle('Matsubara GF $t_{{ab}}/D={}$ $\\beta D={}$'.format(tp, BETA))


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
