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

from pytriqs.archive import HDFArchive
from pytriqs.gf.local import GfReFreq, iOmega_n

from multiprocessing import Pool
import numpy as np

# Matsubara interacting self-consistency
def loop_u(urange, tab, t, beta, imet):
    w_n = gf.matsubara_freq(beta, 1025)
    S = Dimer_Solver(U=0, beta=beta, n_points=len(w_n))
    gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, 0, tab)

    S.setup.update({'t': t, 'tab': tab, 'beta': beta})
    file_label = '_uloop_t{t}_tab{tab}_B{beta}.h5'.format(**S.setup)
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


def plot_re(filen, U):
    R = HDFArchive(filen, 'r')
#
    greal = GfReFreq(indices=[1], window=(-4.0, 4.0), n_points=256)
    greal.set_from_pade(R['U{}'.format(U)]['G_iw']['A', 'A'], 100, 0.0)
    oplot(-1*greal, RI='I', label=r'$U={}$'.format(U), num=4)

    del R

#
p = Pool(6)
tabra = np.arange(0, 1.3, 0.1)
ur = np.arange(0, 6, 0.05)
def dim_met(tab):
    return loop_u(ur, tab, 0.5, 150, 'metal')


met = p.map(dim_met, tabra.tolist())
#
#
## insulating
ur = np.arange(4.5, 0, -0.05)
def dim_ins(tab):
    return loop_u(ur, tab, 0.5, 150, 'insulator')

ins = p.map(dim_ins, tabra.tolist())
