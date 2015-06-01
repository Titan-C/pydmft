# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:51:22 2015

@author: oscar
"""

from pytriqs.gf.local import iOmega_n
import dmft.common as gf
import numpy as np
from multiprocessing import Pool
from dmft.RKKY_dimer_IPT import mix_gf_dimer, init_gf_met, init_gf_ins, \
    Dimer_Solver, dimer


# Matsubara interacting self-consistency
def loop_u(urange, tab, t, beta, file_str):
    w_n = gf.matsubara_freq(beta, 1025)
    S = Dimer_Solver(U=0, beta=beta, n_points=len(w_n))
    gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, 0, tab)
    S.setup.update({'t': t, 'tab': tab, 'beta': beta})

    if file_str.startswith('met'):
        init_gf_met(S.g_iw, w_n, 0, tab, t)
    else:
        init_gf_ins(S.g_iw, w_n, 0, tab, urange[0])

    for u_int in urange:
        S.U = u_int
        dimer(S, gmix, file_str, '/U{U}/')

    return True

#

#met = [loop_u(np.arange(0, 6, 0.1), tab, 0.5, 64., 'metal') for tab in tabra]

#ins = [loop_u(np.arange(6, 0, -0.01), tab, 0.5, 20., 'ins') for tab in tabra]
def dimhelp(tab):
    return loop_u(np.arange(0, 4.5, 0.01), tab, 0.5, 150., 'met_fuloop_t{t}_tab{tab}_B{beta}.h5')


p = Pool(6)
tabra = np.arange(0, 1.3, 0.025)


ou = p.map(dimhelp, tabra.tolist())