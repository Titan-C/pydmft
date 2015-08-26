#!/usr/bin/env python
# Author: KH, March 2007
import dmft.common as gf
import numpy as np
import os,sys,subprocess
import shutil

"""
This module runs ctqmc impurity solver for one-band model.
The executable shoule exist in directory params['exe']
"""


Niter = 2 # Number of DMFT iterations
Uc=2.4
beta=100.
M=10e6


params = {"exe"          : ['ctqmc'          , "# Path to executable"],
          "Delta"        : ["Delta.inp"         , "# Input bath function hybridization"],
          "cix"          : ["one_band.imp"      , "# Input file with atomic state"],
          "U"            : [Uc                  , "# Coulomb repulsion (F0)"],
          "mu"           : [Uc/2.               , "# Chemical potential"],
          "beta"         : [beta                , "# Inverse temperature"],
          "M"            : [M                   , "# Number of Monte Carlo steps"],
          "nom"          : [beta                , "# number of Matsubara frequency points to sample"],
          "nomD"         : [0                   , "# number of Matsubara points using the Dyson Equation"],
          "Segment"      : [0                   , "# Whether to use segment type algorithm"],
          "aom"          : [5                   , "# number of frequency points to determin high frequency tail"],
          "tsample"      : [30                  , "# how often to record the measurements" ],
          "PChangeOrder" : [0.9                 , "# Ratio between trial steps: add-remove-a-kink / move-a-kink"],
          "OCA_G"        : [False               , "# No OCA diagrams being computed - for speed"],
          "minM"         : [1e-10               , "# The smallest allowed value for the atomic trace"],
          "minD"         : [1e-10               , "# The smallest allowed value for the determinant"],
          "Nmax"         : [beta*2              , "# Maximum perturbation order allowed"],
          "GlobalFlip"   : [100000              , "# Global flip shold be tried"],

}

icix="""# Cix file for cluster DMFT with CTQMC
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
    """This subroutine creates Delta.inp from Gf.out for DMFT on bethe lattice: Delta=t^2*G
    If Gf.out does not exist, it creates Gf.out which corresponds to the non-interacting model
    In the latter case also creates the inpurity cix file, which contains information about
    the atomic states.
    """
    fileGf = 'Gf.out'
    try:
        Gf = np.loadtxt(fileGf).T# If output file exists, start from previous iteration
    except: # otherwise start from non-interacting limit
        print 'Starting from non-interacting model'
        w_n = gf.matsubara_freq(params['beta'][0])
        Gf = gf.greenF(w_n)
        Gf = np.array([w_n, Gf.real, Gf.imag])


        # creating impurity cix file
        with open(params['cix'][0], 'w') as f:
            f.write(icix)

    # Preparing input file Delta.inp
    delta = np.array([Gf[0], 0.25*Gf[1], 0.25*Gf[2]]).T
    np.savetxt(fDelta, delta)


# Creating parameters file PARAMS for qmc execution
CreateInputFile(params)

mpifile = 'mpi_prefix.dat'
if os.path.isfile(mpifile):
    mpi_prefix = open(mpifile, 'r').next().strip()
    print "DmftEnvironment: mpi_prefix.dat exists -- running in parallel mode."
else:
    print "DmftEnvironment: mpi_prefix.dat does not exists -- running in serial mode."
    mpi_prefix = ''

fh_info = open('info.dat', 'w')

for it in range(Niter):
    # Constructing bath Delta.inp from Green's function
    DMFT_SCC(params['Delta'][0])

    # Running ctqmc
    print 'Running ---- qmc itt.: ', it, '-----'
    #print os.popen(params['exe'][0]).read()

    cmd = mpi_prefix+' '+params['exe'][0]+'  PARAMS > nohup_imp.out 2>&1 '
    subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)
    fh_info.flush()

    # Some copying to store data obtained so far (at each iteration)
    shutil.copy('Gf.out', 'Gf.out.'+str(it))
    shutil.copy('Sig.out', 'Sig.out.'+str(it))
