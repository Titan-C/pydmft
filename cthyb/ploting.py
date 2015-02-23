# -*- coding: utf-8 -*-
"""
ploting the data
"""

import matplotlib.pyplot as plt
import numpy as np
from dmft.common import matsubara_freq, greenF, gw_invfouriertrans, gt_fouriertrans

parms = {
    'N_TAU'               : 1000,
    'N_MATSUBARA'         : 200,
    'BETA'                : 45
}

iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

g_tau = np.loadtxt('Gt.dat')

fig_gt = plt.figure()
for i in range(1,3):
    plt.plot(tau, g_tau[:, i], label=str(i))
plt.legend(loc=0)
plt.ylabel(r'$G(\tau)$')
plt.xlabel(r'$\tau$')
plt.title('imaginary time green function')
fig_gt.savefig('G_tau.png', format='png',
               transparent=False, bbox_inches='tight', pad_inches=0.05)

fig_gw = plt.figure()
for i in range(1,3):
    gw = gt_fouriertrans(g_tau[:, i], tau, iwn, parms['BETA'])
    plt.plot(iwn.imag, gw.real, '+-', label='Re ' + str(i))
    plt.plot(iwn.imag, gw.imag, 's-', label='Im ' + str(i))
plt.plot(iwn.imag, (1/iwn).imag, '-', label='high w tail ')
plt.ylim([-0.56, 0.05])
plt.legend(loc=0)
plt.ylabel(r'$G(i\omega_n)$')
plt.xlabel(r'$i\omega_n$')
plt.title('Matusubara frequency green function')
fig_gw.savefig('G_iwn.png', format='png',
               transparent=False, bbox_inches='tight', pad_inches=0.05)


f_tau = np.loadtxt('Ft.dat')

order = np.loadtxt('orders.dat')
fig_ord = plt.figure()
plt.plot(order[:,0],order[:,1], label='expansion order')
plt.legend(loc=0)
fig_ord.savefig('orders.png', format='png',
               transparent=False, bbox_inches='tight', pad_inches=0.05)

filesofg=['Gt_0.dat','Gt_1.dat','Gt_2.dat','Gt_3.dat', 'Gt.dat']
for fil in filesofg:

  g_tau = np.loadtxt(fil)


  fig_gt = plt.figure()
  for i in range(1,3):
    plt.plot(tau, g_tau[:, i], label=str(i))
  plt.legend(loc=0)
  plt.ylabel(r'$G(\tau)$')
  plt.xlabel(r'$\tau$')
  plt.title('imaginary time green function {}'.format(fil))
  fig_gt.savefig('G_tau.png', format='png',
               transparent=False, bbox_inches='tight', pad_inches=0.05)

  fig_gw = plt.figure()
  for i in range(1,3):
    gw = gt_fouriertrans(g_tau[:, i], tau, iwn, parms['BETA'])
    plt.plot(iwn.imag, gw.real, '+-', label='Re ' + str(i))
    plt.plot(iwn.imag, gw.imag, 's-', label='Im ' + str(i))
  plt.plot(iwn.imag, (1/iwn).imag, '-', label='high w tail ')
  plt.ylim([-0.56, 0.05])
  plt.legend(loc=0)
  plt.ylabel(r'$G(i\omega_n)$')
  plt.xlabel(r'$i\omega_n$')
  plt.title('Matusubara frequency green function {}'.format(fil))
  fig_gw.savefig('G_iwn.png', format='png',
               transparent=False, bbox_inches='tight', pad_inches=0.05)


  f_tau = np.loadtxt('Ft.dat')

  order = np.loadtxt('orders.dat')
  fig_ord = plt.figure()
  plt.plot(order[:,0],order[:,1], label='expansion order {}'.format(fil))
  plt.legend(loc=0)
  fig_ord.savefig('orders.png', format='png',
               transparent=False, bbox_inches='tight', pad_inches=0.05)
