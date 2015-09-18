#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dimer in Bethe lattice
======================

"""

from math import sqrt, modf
from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import inverse, iOmega_n, SemiCircular
from pytriqs.operators import c, dagger
from time import time
import argparse
import pytriqs.utility.mpi as mpi


def prepare_interaction(u_int):
    """Build the local interaction term of the dimer

    using the symmetric anti-symmetric basis"""

    aup = (-c('low_up', 0) + c('high_up', 0))/sqrt(2)
    adw = (-c('low_dw', 0) + c('high_dw', 0))/sqrt(2)

    bup = (c('low_up', 0) + c('high_up', 0))/sqrt(2)
    bdw = (c('low_dw', 0) + c('high_dw', 0))/sqrt(2)

    h_int = u_int * (dagger(aup)*aup*dagger(adw)*adw +
                     dagger(bup)*bup*dagger(bdw)*bdw)
    return h_int


def set_new_seed(setup):
    """Generates a new starting Green's function for the DMFT loop
    based on the finishing state of the system at a diffent parameter set"""

    src_U = 'U' + str(setup['new_seed'][0])
    dest_U = 'U' + str(setup['new_seed'][1])
    avg_over = int(setup['new_seed'][2])

    with HDFArchive(setup['ofile'].format(**SETUP), 'a') as outp:
        last_iterations = outp[src_U].keys()[-avg_over:]
        giw = pss.averager(outp[src_U], last_iterations)
        try:
            dest_count = len(outp[dest_U].keys())
        except KeyError:
            dest_count = 0
        dest_group = '/{}/it{:0>2}/'.format(dest_U, dest_count)

        outp[dest_group + 'giw'] = giw
        outp[dest_group + 'setup'] = outp[src_U]['it00']['setup']

    print(setup['new_seed'])


def dmft_loop(setup):
    """Starts impurity solver with DMFT paramagnetic self-consistency"""

    if setup['new_seed']:
        set_new_seed(setup)
        return

    imp_sol = Solver(beta=setup['BETA'],
                     gf_struct={'low_up': [0], 'high_up': [0],
                                'low_dw': [0], 'high_dw': [0]})
    h_int = prepare_interaction(setup['U'])

    src_U = 'U'+str(setup['U'])

    try:
        with HDFArchive(setup['ofile'].format(**setup), 'a') as outp:
            last_loop = len(outp[src_U].keys())
            last_it = 'it{:03}'.format(last_loop-1)
            imp_sol.G_iw = outp[src_U][last_it]['G_iw']
    except (KeyError, IOError):
        for name, gblock in imp_sol.G_iw:
            gblock << SemiCircular(1)

    for loop in range(last_loop, last_loop + setup['Niter']):
        if mpi.is_master_node(): print('\n\n in loop \n', '='*40, loop)

        imp_sol.G_iw['low_up'] << 0.5 * (imp_sol.G_iw['low_up'] +
                                         imp_sol.G_iw['low_dw'])
        imp_sol.G_iw['high_up'] << 0.5 * (imp_sol.G_iw['high_up'] +
                                          imp_sol.G_iw['high_dw'])

        imp_sol.G_iw['low_dw'] << imp_sol.G_iw['low_up']
        imp_sol.G_iw['high_dw'] << imp_sol.G_iw['high_up']

        for name, g0block in imp_sol.G0_iw:
            shift = 1. if 'high' in name else -1
            g0block << inverse(iOmega_n + setup['U']/2. + shift * setup['tp'] -
                               0.25*imp_sol.G_iw[name])

        imp_sol.solve(h_int=h_int, **setup['s_params'])

        if mpi.is_master_node():
            with HDFArchive(setup['ofile'].format(**setup)) as last_run:
                last_run['/U{}/it{:03}/G_iw'.format(setup['U'], loop)] = imp_sol.G_iw
                last_run['/U{}/it{:03}/setup'.format(setup['U'], loop)] = setup


def do_setup():
    """Set the solver parameters"""

    parser = argparse.ArgumentParser(description='DMFT loop for CTHYB dimer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sweeps', metavar='MCS', type=float, default=int(1e6),
                        help='Number MonteCarlo Measurement')
    parser.add_argument('-therm', type=int, default=int(1e4),
                        help='Monte Carlo sweeps of thermalization')
    parser.add_argument('-N_meas', type=int, default=200,
                        help='Number of Updates before measurements')
    parser.add_argument('-Niter', metavar='N', type=int,
                        default=20, help='Number of iterations')
    parser.add_argument('-BETA', metavar='B', type=float,
                        default=32., help='The inverse temperature')
    parser.add_argument('-U', type=float,
                        default=2.5, help='Local interaction strenght')
    parser.add_argument('-tp', default=0.18, type=float,
                        help='The dimerization strength')
    parser.add_argument('-ofile', default='DIMER_PM_B{BETA}_tp{tp}.h5',
                        help='Output file shelve')

    parser.add_argument('-new_seed', type=float, nargs=3, default=False,
                        metavar=('U_src', 'U_target', 'avg_over'),
                        help='Resume DMFT loops from on disk data files')
    setup = vars(parser.parse_args())

    fracp, intp = modf(time())
    setup.update({'s_params': {'move_double': True,
                               'n_cycles': int(setup['sweeps']),
                               'n_warmup_cycles': setup['therm'],
                               'length_cycle': setup['N_meas'],
                               'measure_pert_order': True,
                               'random_seed': int(intp+mpi.rank*341*fracp)}})

    return setup

if __name__ == "__main__":
    SETUP = do_setup()

    dmft_loop(SETUP)
