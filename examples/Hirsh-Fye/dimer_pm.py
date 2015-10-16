#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
============================================
QMC Hirsch - Fye Impurity solver for a Dimer
============================================

To treat the dimer in a Bethe lattice and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm
"""

from __future__ import division, absolute_import, print_function


from mpi4py import MPI
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import iOmega_n
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.hirschfye as hf
import numpy as np
import sys

comm = MPI.COMM_WORLD


def averager(it_output, last_iterations):
    """Averages over the files terminating with the numbers given in vector"""
    sgiwd = 0
    sgiwo = 0
    for step in last_iterations:
        sgiwd += it_output[step]['G_iwd']
        sgiwo += it_output[step]['G_iwo']

    sgiwd *= 1./len(last_iterations)
    sgiwo *= 1./len(last_iterations)

    return sgiwd, sgiwo


def set_new_seed(setup):
    """Generates a new starting Green's function for the DMFT loop
    based on the finishing state of the system at a diffent parameter set"""

    src_U = 'U' + str(setup['new_seed'][0])
    dest_U = 'U' + str(setup['new_seed'][1])
    avg_over = int(setup['new_seed'][2])

    with HDFArchive(setup['ofile'].format(**SETUP), 'a') as outp:
        last_iterations = outp[src_U].keys()[-avg_over:]
        giwd, giwo = averager(outp[src_U], last_iterations)
        try:
            dest_count = len(outp[dest_U].keys())
        except KeyError:
            dest_count = 0
        dest_group = '/{}/it{:03}/'.format(dest_U, dest_count)

        outp[dest_group + 'setup/'] = outp[src_U][last_iterations[-1]]['setup']
        outp[dest_group + 'G_iwd/'] = giwd
        outp[dest_group + 'G_iwo/'] = giwo

    print(setup['new_seed'])

class Dimer_Solver_hf(Dimer_Solver):

    def __init__(self, **params):
        super(Dimer_Solver_hf, self).__init__(**params)
        self.g0_tau = GfImTime(indices=['A', 'B'], beta=self.beta,
                               n_points=self.setup['N_TAU'])
        self.g_tau = self.g0_tau.copy()
        self.intm = hf.interaction_matrix(params['BANDS'])
        self.tau, self.w_n = gf.tau_wn_setup(params)

    def solve(self):

        self.g0_tau << InverseFourier(self.g0_iw)
        g0t = np.rollaxis(np.asarray([self.g0_tau(t).real for t in self.tau]), 0, 3)

        gtu, gtd = hf.imp_solver([g0t]*2, self.V_field, self.intm, self.setup)
        gt_D = -0.25 * (gtu[0, 0] + gtu[1, 1] + gtd[0, 0] + gtd[1, 1])
        gt_N = -0.25 * (gtu[1, 0] + gtu[0, 1] + gtd[1, 0] + gtd[0, 1])

        giw_D = gf.gt_fouriertrans(gt_D, self.tau, self.w_n)
        giw_N = gf.gt_fouriertrans(gt_D, self.tau, self.w_n)

        load_gf_from_np(self.g_iw, giw_D, giw_N)


def init_gf_met(omega, mu, tab, tn, t):
    """Gives a metalic seed of a non-interacting system

    """
    G1 = gf.greenF(omega, mu=mu-tab, D=2*(t+tn))
    G2 = gf.greenF(omega, mu=mu+tab, D=2*abs(t-tn))
    return .5*(G1 + G2), .5*(G1 - G2)


def dmft_loop_pm(simulation):
    """Implementation of the solver"""
    setup = {'N_TAU':     2**12,
             't':         0.5,
             'MU':        0.,
             'BANDS':     1,
             'SITES':     2,
             'save_logs': False,
             'global_flip': True,
             }

    if simulation['new_seed']:
        if comm.rank == 0:
            set_new_seed(simulation)
        simulation['U'] = simulation['new_seed'][1]
        return

    setup.update(simulation)
    setup['dtau_mc'] = setup['BETA']/2./setup['N_MATSUBARA']
    current_u = 'U'+str(setup['U'])

    tau, w_n = gf.tau_wn_setup(setup)
    setup['n_tau_mc'] = len(tau)
    giw_D, giw_N = init_gf_met(w_n, setup['MU'], setup['tp'], 0., 0.5)

    try:  # try reloading data from disk
        with HDFArchive(setup['ofile'].format(**setup), 'r') as last_run:
            last_loop = len(last_run[current_u].keys())
            last_it = 'it{:03}'.format(last_loop-1)
            giw_D = last_run[current_u][last_it]['G_iwd']
            giw_N = last_run[current_u][last_it]['G_iwo']
    except (IOError, KeyError):  # if no data clean start
        last_loop = 0


    V_field = hf.ising_v(setup['dtau_mc'], setup['U'],
                           L=setup['SITES']*setup['n_tau_mc'])

    for loop_count in range(last_loop, last_loop + setup['Niter']):
        if comm.rank == 0:
            print('On loop', loop_count, 'beta', setup['BETA'],
                  'U', setup['U'], 'tp', setup['tp'])


        # Bethe lattice bath
        giw0_D = w_n - 0.25 * giw_D
        giw0_N = -setup['tp'] - 0.25 * giw_N

        det = giw_D**2 - giw_N**2
        giw0_D /= det
        giw0_N /= det
        solve()

        if comm.rank == 0:
            with HDFArchive(setup['ofile'].format(**setup), 'a') as simulation:
                simulation[current_u+'/it{:03}'.format(loop_count)] = {
                            'setup':  S.setup.copy(),
                            'G_iwd':  S.g_iw['A', 'A'],
                            'G_iwo':  S.g_iw['A', 'B'],
                            }
        sys.stdout.flush()

if __name__ == "__main__":
    parser = hf.do_input('DMFT loop for Hirsh-Fye dimer lattice')
    parser.add_argument('-tp', type=float, default=.25,
                        help='Dimerization strength')
    parser.add_argument('-df', '--double_flip_prob', type=float, default=0.,
                        help='Probability for double spin flip on equal sites')

    SETUP = vars(parser.parse_args())
    dmft_loop_pm(SETUP)
