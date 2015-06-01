# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:51:22 2015

@author: oscar
"""

import sys
sys.path.append('/home/oscar/libs/lib/python2.7/site-packages')
import dmft.common as gf
from plot_dimer_bethe_triqs import mix_gf_dimer, init_gf_met, init_gf_ins
from RKKY_dimer_IPT import Dimer_Solver, dimer

from pytriqs.gf.local import iOmega_n

import numpy as np

# Matsubara interacting self-consistency
def loop_u(urange, tab, t, beta, imet):
    w_n = gf.matsubara_freq(beta, 2025)
    S = Dimer_Solver(U=0, beta=beta, n_points=len(w_n))
    gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, 0, tab)

    S.setup.update({'t': t, 'tab': tab, 'beta': beta})
    file_label = '_fuloop_t{t}_tab{tab}_B{beta}.h5'.format(**S.setup)
    filename = None
    if imet == 'metal':
        init_gf_met(S.g_iw, w_n, 0, tab, t)
        filename = 'met'+file_label
    else:
        init_gf_ins(S.g_iw, w_n, 0, tab, urange[0])
        filename = 'ins'+file_label

    for u_int in urange:
        S.U = u_int

        step = '/U{}/'.format(S.U)
        dimer(S, gmix, filename, step)

    return True

#
tabra = np.arange(0, 1.3, 0.05)

met = [loop_u(np.arange(0, 6, 0.01), tab, 0.5, 20., 'metal') for tab in tabra]

#ins = [loop_u(np.arange(6, 0, -0.01), tab, 0.5, 20., 'ins') for tab in tabra]