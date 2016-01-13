#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Óscar Nájera Sept 2015
# Author: KH, March 2007
"""
This module runs ctqmc impurity solver for one-band model.
The executable shoule exist in directory params['exe']
"""

from glob import glob
import argparse
import multiprocessing
import dmft.common as gf
import numpy as np
import os
import dmft.plot.cthyb_h_single_site as psb
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description='DMFT loop for CTHYB single band',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-BETA', metavar='B', type=float,
                    default=64., help='The inverse temperature')
parser.add_argument('-Niter', metavar='N', type=int,
                    default=3, help='Number of iterations')
parser.add_argument('-U', metavar='U', nargs='+', type=float,
                    default=[2.7], help='Local interaction strenght')

parser.add_argument('-odir', default='coex/B{BETA}_U{U}',
                    help='Output directory')
parser.add_argument('-new_seed', type=float, nargs=3, default=False,
                    metavar=('U_src', 'U_target', 'avg_over'),
                    help='Resume DMFT loops from on disk data files')
parser.add_argument('-sweeps', metavar='MCS', type=float, default=int(1e6),
                    help='Number Monte Carlo Measurement')

args = parser.parse_args()
Niter = args.Niter
BETA = args.BETA
M = int(args.sweeps)


params = {"exe":          ['ctqmc',        "# Path to executable"],
          "Delta":        ["Delta.inp",    "# Input bath function hybridization"],
          "cix":          ["one_band.imp", "# Input file with atomic state"],
          "beta":         [BETA,           "# Inverse temperature"],
          "M":            [M,              "# Number of Monte Carlo steps"],
          "nom":          [BETA,           "# number of Matsubara frequency points to sample"],
          "nomD":         [0,              "# number of Matsubara points using the Dyson Equation"],
          "Segment":      [0,              "# Whether to use segment type algorithm"],
          "aom":          [5,              "# number of frequency points to determin high frequency tail"],
          "tsample":      [30,             "# how often to record the measurements" ],
          "PChangeOrder": [0.9,            "# Ratio between trial steps: add-remove-a-kink / move-a-kink"],
          "OCA_G":        [False,          "# No OCA diagrams being computed - for speed"],
          "minM":         [1e-10,          "# The smallest allowed value for the atomic trace"],
          "minD":         [1e-10,          "# The smallest allowed value for the determinant"],
          "Nmax":         [BETA*2,         "# Maximum perturbation order allowed"],
          "GlobalFlip":   [100000,         "# Global flip shold be tried"],
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


def set_new_seed(setup):
    new_seed = setup.new_seed
    avg_over = int(new_seed[2])

    prev_iter = sorted(glob(setup.odir.format(BETA=setup.BETA, U=new_seed[0]) +
                            '/Gf.out.*'))[-avg_over:]
    giw = psb.averager(prev_iter).T

    udir = setup.odir.format(BETA=setup.BETA, U=new_seed[1])
    if not os.path.exists(udir):
        os.makedirs(udir)
    np.savetxt(udir + '/Gf.out', giw)
    # creating impurity cix file
    with open(udir + '/' + params['cix'][0], 'w') as f:
        f.write(icix)


def dmft_loop_pm(Uc):
    """Creating parameters file PARAMS for qmc execution"""
    uparams = {"U": [Uc, "# Coulomb repulsion (F0)"],
               "mu": [Uc/2., "# Chemical potential"]}
    params.update(uparams)
    CreateInputFile(params)

    mpi_prefix = 'mpirun -np ' + str(multiprocessing.cpu_count())

    fh_info = open('info.dat', 'w')

    prev_iter = len(glob('Gf.out.*'))
    print('Loop at beta ', BETA, ' U=', Uc)

    for it in range(prev_iter, prev_iter + Niter):
        # Constructing bath Delta.inp from Green's function
        DMFT_SCC(params['Delta'][0])

        # Running ctqmc
        print('Running ---- qmc it: ', it, '-----')
        sys.stdout.flush()

        cmd = mpi_prefix+' '+params['exe'][0]+'  PARAMS > nohup_imp.out 2>&1 '
        subprocess.call(cmd, shell=True, stdout=fh_info, stderr=fh_info)
        fh_info.flush()

        # Some copying to store data obtained so far (at each iteration)
        shutil.copy('Gf.out', 'Gf.out.{:02}'.format(it))
        shutil.copy('Sig.out', 'Sig.out.{:02}'.format(it))


CWD = os.getcwd()
if args.new_seed:
    set_new_seed(args)
    sys.exit()

for Uc in args.U:

    udir = args.odir.format(BETA=BETA, U=Uc)
    if not os.path.exists(udir):
        os.makedirs(udir)

    os.chdir(udir)
    dmft_loop_pm(Uc)
    os.chdir(CWD)
