# -*- coding: utf-8 -*-
"""
ploting the data
"""

import matplotlib.pyplot as plt
import numpy as np
from dmft.common import matsubara_freq, gw_invfouriertrans, gt_fouriertrans
import sys
sys.path.append('/home/oscar/libs/lib/')
import pyalps
import pyalps.plot
from pyalps.hdf5 import archive




def plot_gt_iter(basename, iteration):
    final = archive(basename + '.out.h5')
    parms = final['parameters']
    steps = archive(basename+'steps.h5', 'r')

    fig_gt = plt.figure()
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    for n in range(iteration):
        gtau = steps['iter_{}/G_tau/'.format(n)]
        for i in range(2):
            plt.semilogy(tau, -gtau[i], label='it {}, sp {}'.format(n, i))
    plt.legend(loc=0)
    plt.ylabel(r'$G(\tau)$')
    plt.xlabel(r'$\tau$')
    plt.title('imaginary time green function iterations')
    fig_gt.savefig('G_tau.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)

    del final
    del steps


def plot_gw_iter(basename, iteration):
    final = archive(basename + '.out.h5')
    parms = final['parameters']
    steps = archive(basename+'steps.h5', 'r')

    fig_gw = plt.figure()
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])

    for n in range(iteration):
        gtau = steps['iter_{}/G_tau/'.format(n)]
        gw = gt_fouriertrans(gtau[0], tau, iwn, parms['BETA'])
        plt.plot(iwn.imag, gw.real, '+-', label='Re ')
        plt.plot(iwn.imag, gw.imag, 's-', label='Im ')
    plt.plot(iwn.imag, (1/iwn).imag, '-', label='high w tail ')
    plt.ylim([-2, 0.05])
    plt.legend(loc=0)
    plt.ylabel(r'$G(i\omega_n)$')
    plt.xlabel(r'$i\omega_n$')
    plt.title('Matusubara frequency green function')
    fig_gw.savefig('G_iwn.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)
    del final
    del steps


def plot_end(filename):
    sim = archive('PM_b50_U2.2.out.h5')
    parms = sim['parameters']
    tau = np.linspace(0, parms['BETA'], 'N_TAU'+1)
    fig_gt = plt.figure()
    for i in range(parms['N_ORBITALS']):
        plt.errorbar(tau, sim['G_tau/{}/mean/value'.format(i)],
                     yerr=sim['G_tau/{}/mean/error'].format(i))
    plt.legend(loc=0)
    plt.ylabel(r'$G(\tau)$')
    plt.xlabel(r'$\tau$')
    plt.title('imaginary time green function beta {} U {}'.format(parms['BETA'], parms['U']))
    fig_gt.savefig('G_tau'+parms['BASENAME']+'.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)

    fig_gw = plt.figure()
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    gw0 = gt_fouriertrans(sim['G_tau/0/mean/value'], tau, iwn, parms['BETA'])
    plt.plot(iwn.imag, gw0.real, '+-', label='RE')
    plt.plot(iwn.imag, gw0.imag, 's-', label='IM')
    plt.legend(loc=0)
    plt.ylabel(r'$G(i\omega_n)$')
    plt.xlabel(r'$i\omega_n$')
    plt.title('Matusubara frequency green function beta {} U {}'.format(parms['BETA'], parms['U']))

    fig_gw.savefig('G_iwn'+parms['BASENAME']+'.png', format='png',
                   transparent=False, bbox_inches='tight', pad_inches=0.05)

    fig_siw = plt.figure()
    sig=iwn + parms['MU'] - gw0 -1/gw0
    plt.plot(iwn.imag, sig.real, '+-')
    plt.plot(iwn.imag, sig.imag, 's-')
    plt.legend(loc=0)
    plt.ylabel(r'$\Sigma(i\omega_n)$')
    plt.xlabel(r'$i\omega_n$')
    plt.title('Matusubara frequency Self-Energy beta {} U {}'.format(parms['BETA'], parms['U']))
    fig_siw.savefig('Sig_iwn'+parms['BASENAME']+'.png', format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)

    del sim
