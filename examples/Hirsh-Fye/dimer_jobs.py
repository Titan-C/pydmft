# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:03:46 2015

@author: oscar
"""
import argparse
import numpy as np
from joblib import Parallel, delayed
from plot_dimer_pm import dmft_loop_pm

parser = argparse.ArgumentParser(description='DMFT loop for a dimer bethe\
                                              lattice solved by IPT')
parser.add_argument('dt', metavar='dt', type=float,
                    default=.5, help='The inverse temperature')

parser.add_argument('beta', metavar='B', type=float,
                    default=16., help='The inverse temperature')

args = parser.parse_args()

PARAMS = {'N_TAU':    2**13,
          'dtau_mc': args.dt,
          'U':           0.,
          'MU':          0.,
          'BANDS': 1,
          'SITES': 2,
          'loops':       0,  # starting loop count
          'max_loops':   20,
          'sweeps':      int(2e6),
          'therm':       int(1e5),
          'N_meas':      3,
          'save_logs':   False,
          'updater':     'discrete',
          'convegence_tol': 4e-3,
          }

TABRA = np.arange(0.18, 0.3, 0.01)
BETA = args.beta

ur = np.arange(2, 3, 0.1)
Parallel(n_jobs=6, verbose=5)(delayed(dmft_loop_pm)(ur,
                             tab, 0.5, 0., BETA,
                             'disk/metf_HF_Ul_tp{tp}_B{BETA}.h5')
                             for tab in TABRA)
