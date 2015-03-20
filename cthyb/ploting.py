# -*- coding: utf-8 -*-
"""
ploting the data
"""

import matplotlib.pyplot as plt
import numpy as np
from dmft.common import matsubara_freq, gt_fouriertrans
from cthyb.storage import recover_measurement
import h5py


import pyalps
import pyalps.plot
from pyalps.hdf5 import archive


def open_iterations(basename):
    final = archive(basename + '.out.h5')
    parms = final['parameters']
    del final

    return parms, h5py.File(basename+'steps.h5', 'r')



def plot_gt_iter(basename, orb):
    parms, steps = open_iterations(basename)

    fig_gt = plt.figure()
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    for it in sorted(steps):
        gtau = steps[it+'/G_tau/'+str(orb)].value
        plt.semilogy(tau, -gtau, label=it)
    plt.legend(loc=0)
    plt.ylabel(r'$G(\tau) sp{}$'.format(orb))
    plt.xlabel(r'$\tau$')
    plt.title(r'DMFT Iterations of $G(\tau)$ at $\beta= {}$, $U= {}$'.format(parms['BETA'], parms['U']))
    fig_gt.savefig('G_tau.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)

    del steps


def plot_gw_iter(basename, orb):
    parms, steps = open_iterations(basename)

    fig_gw, (ax_re, ax_im) = plt.subplots(2, sharex=True)
    fig_gw.subplots_adjust(hspace=0)
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])

    for it in sorted(steps):
        gtau = steps[it+'/G_tau/'+str(orb)].value
        gw = gt_fouriertrans(gtau, tau, iwn)
        ax_re.plot(iwn.imag, gw.real, '+-', label=it)
        ax_im.plot(iwn.imag, gw.imag, 's-', label=it)
    plt.legend(loc=4)
    ax_re.set_ylabel(r'$\Re G(i\omega_n) sp{}$'.format(orb))
    ax_im.set_ylabel(r'$\Im G(i\omega_n) sp{}$'.format(orb))
    plt.xlim([0, 4.5])
    plt.xlabel(r'$i\omega_n$')
    ax_re.set_title(r'DMFT Iterations of $G(i\omega_n)$ at $\beta= {}$, $U= {}$'.format(parms['BETA'], parms['U']))
    fig_gw.savefig('G_iwn.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)
    del steps


def plot_end(filename):
    sim = archive(filename)
    parms = sim['parameters']
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    fig_gt = plt.figure()
    for i in range(parms['N_ORBITALS']):
        plt.errorbar(tau, sim['G_tau/{}/mean/value'.format(i)],
                     yerr=sim['G_tau/{}/mean/error'.format(i)], label='spin{}'.format(i))
    plt.legend(loc=0)
    plt.ylabel(r'$G(\tau)$')
    plt.xlabel(r'$\tau$')
    plt.title(r'$G(\tau)$ at $\beta= {}$, $U= {}$'.format(parms['BETA'], parms['U']))
    fig_gt.savefig('G_tau'+parms['BASENAME']+'.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)

    fig_gw, gw_ax = plt.subplots()
    fig_siw, sw_ax = plt.subplots()
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    for i in range(parms['N_ORBITALS']):
        try:
            gw = sim['G_omega/{}/mean/value'.format(i)]
        except:
            gw = gt_fouriertrans(sim['G_tau/{}/mean/value'.format(i)], tau, iwn)
        gw_ax.plot(iwn.imag, gw.real, '+-', label='RE, sp{}'.format(i))
        gw_ax.plot(iwn.imag, gw.imag, 's-', label='IM, sp{}'.format(i))

        sig = iwn + parms['MU'] - gw - 1/gw
        sw_ax.plot(iwn.imag, sig.real, '+-', label='RE, sp{}'.format(i))
        sw_ax.plot(iwn.imag, sig.imag, 's-', label='IM, sp{}'.format(i))

    # PM AVG
    gt_pm = recover_measurement(parms, 'G_tau').mean(axis=0)
    gw_pm = gt_fouriertrans(gt_pm, tau, iwn)
    gw_ax.plot(iwn.imag, gw_pm.real, '+-', label='RE, pm')
    gw_ax.plot(iwn.imag, gw_pm.imag, 's-', label='IM, pm')
    sig = iwn + parms['MU'] - gw_pm - 1/gw_pm
    sw_ax.plot(iwn.imag, sig.real, '+-', label='RE, pm')
    sw_ax.plot(iwn.imag, sig.imag, 's-', label='IM, pm')

    gw_ax.set_xlim([0, 6.5])
    gw_ax.legend(loc=4)
    gw_ax.set_ylabel(r'$G(i\omega_n)$')
    gw_ax.set_xlabel(r'$i\omega_n$')
    gw_ax.set_title(r'$G(i\omega_n)$ at $\beta= {}$, $U= {}$'.format(parms['BETA'], parms['U']))

    fig_gw.savefig('G_iwn'+parms['BASENAME']+'.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)

    plt.xlim([0, 6.5])
    plt.legend(loc=4)
    plt.ylabel(r'$\Sigma(i\omega_n)$')
    plt.xlabel(r'$i\omega_n$')
    plt.title(r'$\Sigma(i\omega_n)$ at $\beta= {}$, $U= {}$'.format(parms['BETA'], parms['U']))
    fig_siw.savefig('Sig_iwn'+parms['BASENAME']+'.png', format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)

    del sim


def plot_order(filename):
    sim = archive(filename)
    parms = sim['parameters']
    fig_order = plt.figure()
    hist = 'simulation/results/order_histogram_'
    for i in range(parms['N_ORBITALS']):
        plt.errorbar(np.arange(parms['N_HISTOGRAM_ORDERS']), sim[hist + '{}/mean/value'.format(i)],
                     yerr=sim[hist + '{}/mean/error'.format(i)], label='spin{}'.format(i))

    plt.title(r'Expansion order at $\beta= {}$, $U= {}$'.format(parms['BETA'], parms['U']) + '\nwith {} MC sweeps and avg Sign {}'.format(sim['simulation/results/Sign/count'], sim['simulation/results/Sign/mean/value']))
    plt.legend(loc=0)


    del sim
