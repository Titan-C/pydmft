#!/usr/bin/env python
# Author: KH, March 2007
import dmft.common as gf
import numpy as np
import os,sys,subprocess
import shutil
import argparse

"""
This module runs ctqmc impurity solver for one-band model.
The executable shoule exist in directory params['exe']
"""
eua

parser = argparse.ArgumentParser(description='DMFT loop for CTHYB single band')
parser.add_argument('-beta', metavar='B', type=float,
                    default=64., help='The inverse temperature')
parser.add_argument('-Niter', metavar='N', type=int,
                    default=10, help='Number of iterations')
parser.add_argument('-U', metavar='U', nargs='+', type=float,
                    default=[2.7], help='Local interaction strenght')

args = parser.parse_args()

Niter = args.Niter
Uc = args.U
beta = args.beta
M=10e6


params = {"exe"          : ['ctqmc'          , "# Path to executable"],
          "Delta"        : ["Delta.inp"         , "# Input bath function hybridization"],
          "cix"          : ["link_actqmc.imp"      , "# Input file with atomic state"],
#          "U"            : [Uc                  , "# Coulomb repulsion (F0)"],
#          "mu"           : [Uc/2.               , "# Chemical potential"],
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

    # Preparing input file Delta.inp
    delta = np.array([Gf[0], 0.25*Gf[1], 0.25*Gf[2], 0.25*Gf[1], 0.25*Gf[2]]).T
    np.savetxt(fDelta, delta)



def dmft_loop_pm(Uc):
    # Creating parameters file PARAMS for qmc execution
    uparams = {"U"            : [Uc                  , "# Coulomb repulsion (F0)"],
              "mu"           : [Uc/2.               , "# Chemical potential"]}
    params.update(uparams)
    CreateInputFile(params)

    mpi_prefix = 'mpirun -np 12'

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

cwd = os.getcwd()
for Uc in args.U:

    #udir = 'B{}_U{}'.format(args.beta, Uc)
    #os.makedirs(udir)
    #os.chdir(udir)
    #if os.path.exists('Gf.out'):
        #shutil.copy('Gf.out', udir)
    dmft_loop_pm(Uc)
    #shutil.copy('Gf.out', '../Gf.out')
    #os.chdir(cwd)
