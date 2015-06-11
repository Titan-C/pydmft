# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:51:22 2015

@author: oscar
"""

from pytriqs.gf.local import iOmega_n
import dmft.common as gf
import numpy as np
from dmft.RKKY_dimer_IPT import mix_gf_dimer, init_gf_met, init_gf_ins, \
    Dimer_Solver, dimer
import argparse
from joblib import Parallel, delayed

# Matsubara interacting self-consistency
def loop_u(urange, tab, t, beta, file_str):
    n_freq = int(15.*beta/np.pi)
    w_n = gf.matsubara_freq(beta, n_freq)
    S = Dimer_Solver(U=0, beta=beta, n_points=n_freq)
    gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, 0, tab)
    S.setup.update({'t': t, 'tab': tab, 'beta': beta, 'n_freq': n_freq})

    if file_str.startswith('met'):
        init_gf_met(S.g_iw, w_n, 0, tab, t)
    else:
        init_gf_ins(S.g_iw, w_n, 0, tab, urange[0])

    for u_int in urange:
        S.U = u_int
        dimer(S, gmix, file_str, '/U{U}/')

    return True

# Matsubara interacting self-consistency
def loop_tab(uint, tabr, t, beta, file_str):
    n_freq = int(15.*beta/np.pi)
    w_n = gf.matsubara_freq(beta, n_freq)
    S = Dimer_Solver(U=uint, beta=beta, n_points=n_freq)
    S.setup.update({'t': t, 'U': uint, 'beta': beta, 'n_freq': n_freq})

    init_gf_ins(S.g_iw, w_n, 0, tabr[0], uint)

    for tab in tabr:
        S.setup['tab'] = tab
        gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, 0, tab)
        dimer(S, gmix, file_str, '/tab{tab}/')

    return True

parser = argparse.ArgumentParser(description='DMFT loop for a dimer bethe lattice solved by IPT')
parser.add_argument('beta', metavar='B', type=float,
                    default=150., help='The inverse temperature')


tabra = np.hstack((np.arange(0, 0.5, 0.01), np.arange(0.5, 1.3, 0.025)))
args = parser.parse_args()
BETA = args.beta

ur = np.arange(0, 4.5, 0.025)

#print(BETA)
#Parallel(n_jobs=2, verbose=5)(delayed(loop_u)(ur,
#         tab, 0.5, BETA, 'met_fuloop_t{t}_tab{tab}_B{beta}.h5')
#         for tab in [0.1, 0.2])
#Parallel(n_jobs=-1, verbose=5)(delayed(loop_u)(ur[::-1],
#         tab, 0.5, BETA, 'ins_nm_fuloop_t{t}_tab{tab}_B{beta}.h5')
#         for tab in tabra)

tabra = np.arange(0, 0.5, 0.01)[::-1]
#Parallel(n_jobs=-1, verbose=5)(delayed(loop_tab)(u,
#         tabra, 0.5, BETA, 'ins_tloop_t{t}_U{U}_B{beta}.h5')
#         for u in np.arange(2, 4.5, 0.01))
