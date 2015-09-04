#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Óscar Nájera Sept 2015
# Author: KH, March 2007
"""
This module runs ctqmc impurity solver for one-band model.
The executable shoule exist in directory params['exe']
"""

import argparse
import dmft.common as gf
import glob
import numpy as np
import os
import shutil
import subprocess
import plot_single_band as psb

parser = argparse.ArgumentParser(description='DMFT loop for CTHYB single band')
parser.add_argument('-beta', metavar='B', type=float,
                    default=64., help='The inverse temperature')
parser.add_argument('-Niter', metavar='N', type=int,
                    default=10, help='Number of iterations')
parser.add_argument('-U', metavar='U', nargs='+', type=float,
                    default=[2.7], help='Local interaction strenght')
parser.add_argument('-r', '--resume', action='store_true',
                    help='Resume DMFT loops from inside folder. Do not copy'
                    'a seed file from the main directory')
parser.add_argument('-liter', metavar='N', type=int,
                    default=5, help='On resume, average over liter[ations]')
args = parser.parse_args()

Niter = args.Niter
BETA = args.beta
M = 10e6


params = {"exe":           ['ctqmc',              "# Path to executable"],
          "Delta":         ["Delta.inp"         , "# Input bath function hybridization"],
          "cix":           ["one_band.imp"      , "# Input file with atomic state"],
#          "U"            : [Uc                  , "# Coulomb repulsion (F0)"],
#          "mu"           : [Uc/2.               , "# Chemical potential"],
          "beta"         : [BETA                , "# Inverse temperature"],
          "M"            : [M                   , "# Number of Monte Carlo steps"],
          "nom"          : [BETA                , "# number of Matsubara frequency points to sample"],
          "nomD"         : [0                   , "# number of Matsubara points using the Dyson Equation"],
          "Segment"      : [0                   , "# Whether to use segment type algorithm"],
          "aom"          : [5                   , "# number of frequency points to determin high frequency tail"],
          "tsample"      : [30                  , "# how often to record the measurements" ],
          "PChangeOrder" : [0.9                 , "# Ratio between trial steps: add-remove-a-kink / move-a-kink"],
          "OCA_G"        : [False               , "# No OCA diagrams being computed - for speed"],
          "minM"         : [1e-10               , "# The smallest allowed value for the atomic trace"],
          "minD"         : [1e-10               , "# The smallest allowed value for the determinant"],
          "Nmax"         : [BETA*2              , "# Maximum perturbation order allowed"],
          "GlobalFlip"   : [100000              , "# Global flip shold be tried"],
          }

icix = """# Cix file for cluster DMFT with CTQMC
# cluster_size, number of states, number of baths, maximum matrix size
1 4 2 1
# baths, dimension, symmetry, global flip
0       1 0 0
1       1 0 0
# cluster energies for unique baths, eps[k]
0 0
#   N   K   Sz size F^{+,dn}, F^{+,up}, Ea  S
1   0   0    0   1   2         3        0   0
2   1   0 -0.5   1   0         4        0   0.5
3   1   0  0.5   1   4         0        0   0.5
4   2   0    0   1   0         0        0   0
# matrix elements
1  2  1  1    1    # start-state,end-state, dim1, dim2, <2|F^{+,dn}|1>
1  3  1  1    1    # start-state,end-state, dim1, dim2, <3|F^{+,up}|1>
2  0  0  0
2  4  1  1   -1    # start-state,end-state, dim1, dim2, <4|F^{+,up}|2>
3  4  1  1    1
3  0  0  0
4  0  0  0
4  0  0  0
HB2                # Hubbard-I is used to determine high-frequency
# Uc = U[m1,m2,m3,m1]-U[m1,m2,m1,m3] ; loops [m1,m2,m3]
0 0
0 0
0 0
0 0
# number of operators needed
0
# Data for HB1
1 4 2 1
#  ind  N   K   Sz size
1  1   0   0    0   1     2  3     0   0
2  2   1   0 -0.5   1     0  4     0   0.5
3  3   1   0  0.5   1     4  0     0   0.5
4  4   2   0    0   1     0  0     0   0
# matrix elements
1  2  1  1    1
1  3  1  1    1
2  0  0  0
2  4  1  1   -1
3  4  1  1    1
3  0  0  0
4  0  0  0
4  0  0  0
"""


def CreateInputFile(params):
    " Creates input file (PARAMS) for CT-QMC solver"
    with open('PARAMS', 'w') as parfile:
        for key, vaule in params.iteritems():
            parfile.write('{}\t{}\t{}\n'.format(key, vaule[0], vaule[1]))


def DMFT_SCC(fDelta):
    """This subroutine creates Delta.inp from Gf.out for DMFT on bethe
    lattice: Delta=t^2*G If Gf.out does not exist, it creates Gf.out
    which corresponds to the non-interacting model In the latter case
    also creates the inpurity cix file, which contains information
    about the atomic states."""
    fileGf = 'Gf.out'
    try:
        Gf = np.loadtxt(fileGf).T
        # If output file exists, start from previous iteration
    except Exception:  # otherwise start from non-interacting limit
        print('Starting from non-interacting model at beta'+str(BETA))
        w_n = gf.matsubara_freq(BETA)
        Gf = gf.greenF(w_n)
        Gf = np.array([w_n, Gf.real, Gf.imag])

        # creating impurity cix file
        with open(params['cix'][0], 'w') as f:
            f.write(icix)

    # Preparing input file Delta.inp
    delta = np.array([Gf[0], 0.25*Gf[1], 0.25*Gf[2]]).T
    np.savetxt(fDelta, delta)


def averager(vector, file_str='Gf.out'):
    """Averages over the files terminating with the numbers given in vector"""
    nvec = [file_str+'.{:02}'.format(it) for it in vector]
    new_gf = psb._averager(nvec).T
    np.savetxt(file_str, new_gf)


def dmft_loop_pm(Uc, liter, resume):
    """Creating parameters file PARAMS for qmc execution"""
    uparams = {"U": [Uc, "# Coulomb repulsion (F0)"],
               "mu": [Uc/2., "# Chemical potential"]}
    params.update(uparams)
    CreateInputFile(params)

    mpi_prefix = 'mpirun -np 12'

    fh_info = open('info.dat', 'w')

    prev_iter = len(glob.glob('Gf.out.*'))
    if resume:
        averager(np.arange(prev_iter - liter, prev_iter))

    for it in range(prev_iter, prev_iter + Niter):
        # Constructing bath Delta.inp from Green's function
        DMFT_SCC(params['Delta'][0])

        # Running ctqmc
        print('Running ---- qmc it: ', it, '-----')

        cmd = mpi_prefix+' '+params['exe'][0]+'  PARAMS > nohup_imp.out 2>&1 '
        subprocess.call(cmd, shell=True, stdout=fh_info, stderr=fh_info)
        fh_info.flush()

        # Some copying to store data obtained so far (at each iteration)
        shutil.copy('Gf.out', 'Gf.out.{:02}'.format(it))
        shutil.copy('Sig.out', 'Sig.out.{:02}'.format(it))


CWD = os.getcwd()
for Uc in args.U:

    udir = 'B{}_U{}'.format(BETA, Uc)
    if not os.path.exists(udir):
        os.makedirs(udir)
    seedGF = 'Gf.out.B'+str(BETA)
    if os.path.exists(seedGF) and not args.resume:
        shutil.copy(seedGF, udir+'/Gf.out')
        shutil.copy(params['cix'][0], udir)
    os.chdir(udir)
    dmft_loop_pm(Uc, args.liter, args.resume)
    shutil.copy('Gf.out', '../'+seedGF)
    shutil.copy(params['cix'][0], '../')
    os.chdir(CWD)
