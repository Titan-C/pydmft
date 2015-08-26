# -*- coding: utf-8 -*-
"""
Data analysis finding the coexisting region
"""

import cthyb.ploting as pt
import matplotlib.pyplot as plt
import numpy as np
from dmft.twosite import matsubara_Z
import glob

results = glob.glob('PM_*.out.h5')

for data in results:
    pt.plot_end(data)
    plt.close('all')

def quas(filename):
    sim = pt.archive(filename)
    parms = sim['parameters']
    tau, wn = pt.tau_wn_setup(parms)
    gt_pm = pt.recover_measurement(parms, 'G_tau').mean(axis=0)
    gw_pm = pt.gt_fouriertrans(gt_pm, tau, wn)
    sig = 1j*wn + parms['MU'] - gw_pm - 1/gw_pm
    return matsubara_Z(sig.imag, parms['BETA'])
zet = []


BETA = np.array([100.0, 12.0, 8.0, 18.0, 25.0, 4.0, 50.0, 75.0, 8.0])
U = np.concatenate((np.arange(3.8, 6.3, 0.1), np.arange(6.25, 3.8, -0.1)))
for beta in BETA:
    zet_u = []
    for u_int in U:

        zet_u.append(quas('PM_HF_b{}_U{}.out.h5'.format(beta, u_int)))
    plt.plot(U, zet_u, label=r'$\beta$={}'.format(beta))
    zet.append(np.asarray(zet_u))
