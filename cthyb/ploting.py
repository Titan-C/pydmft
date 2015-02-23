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
            plt.plot(tau, gtau[i], label='it {}, sp {}'.format(n, i))
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
