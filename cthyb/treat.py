# -*- coding: utf-8 -*-
"""
Data analysis finding the coexisting region
"""

import cthyb.ploting as pt
import matplotlib.pyplot as plt
import numpy as np
from dmft.twosite import matsubara_Z
import glob

results = glob.glob('mi/PM_*.out.h5')

for data in  results:
    pt.plot_end(data)
    pt.plt.close('all')

zet = []
for u_int in  np.arange(1, 6.5, 0.2):
    sim = pt.archive('PM_MI_b{}_U{}.out.h5'.format(20.0, u_int))
    parms = sim['parameters']
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    iwn = pt.matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    gt_pm = pt.recover_measurement(parms, 'G_tau').mean(axis=0)
    gw_pm = pt.gt_fouriertrans(gt_pm, tau, iwn)
    sig = iwn + parms['MU'] - gw_pm - 1/gw_pm
    zet.append(matsubara_Z(sig.imag, parms['BETA']))

