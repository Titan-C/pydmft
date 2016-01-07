#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dimer in Bethe lattice
======================

"""

from __future__ import division, absolute_import, print_function
from math import sqrt, modf
from time import time
import argparse
import os
import struct
import dmft.plot.triqs_dimer as tdimer
from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import inverse, iOmega_n, SemiCircular
from pytriqs.operators import c, dagger
import pytriqs.utility.mpi as mpi


def prepare_interaction(u_int):
    """Build the local interaction term of the dimer

    using the symmetric anti-symmetric basis"""

    aup = 1/sqrt(2)*(-c('asym_up', 0) + c('sym_up', 0))
    adw = 1/sqrt(2)*(-c('asym_dw', 0) + c('sym_dw', 0))

    bup = 1/sqrt(2)*(c('asym_up', 0) + c('sym_up', 0))
    bdw = 1/sqrt(2)*(c('asym_dw', 0) + c('sym_dw', 0))

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
        giw = tdimer.get_giw(outp[src_U], slice(-avg_over, None))
        try:
            dest_count = len(outp[dest_U])
        except KeyError:
            dest_count = 0
        dest_group = '/{}/it{:0>2}/'.format(dest_U, dest_count)

        outp[dest_group + 'giw'] = giw

    print(setup['new_seed'])


def dmft_loop(setup, u_int, G_iw):
    """Starts impurity solver with DMFT paramagnetic self-consistency"""

    if setup['new_seed']:
        set_new_seed(setup)
        return

    imp_sol = Solver(beta=setup['BETA'],
                     gf_struct={'asym_up': [0], 'sym_up': [0],
                                'asym_dw': [0], 'sym_dw': [0]})
    h_int = prepare_interaction(u_int)

    src_U = 'U'+str(u_int)

    try:
        with HDFArchive(setup['ofile'].format(**setup), 'r') as outp:
            g_iw_seed = tdimer.get_giw(outp[src_U], slice(-3, None))
            last_loop = len(outp[src_U])
            try:
                imp_sol.G_iw << g_iw_seed
            except IndexError:
                import itertools
                spin = ['up', 'dw']
                newn = (''.join(a) for a in itertools.product(['asym_', 'sym_'], spin))
                oldn = (''.join(a) for a in itertools.product(['high_', 'low_'], spin))
                for name_n, name_o in zip(newn, oldn):
                    imp_sol.G_iw[name_n] << g_iw_seed[name_o]
    except (KeyError, IOError):
        last_loop = 0
        for name, gblock in imp_sol.G_iw:
            gblock << SemiCircular(1)
        if G_iw is not None:
            imp_sol.G_iw << G_iw

    for loop in range(last_loop, last_loop + setup['Niter']):
        if mpi.is_master_node():
            print('\n\n in loop \n', '='*40, loop)

        tdimer.paramagnetic_hf_clean(imp_sol.G_iw, u_int, setup['tp'])

        for name, g0block in imp_sol.G0_iw:
            shift = 1. if 'asym' in name else -1
            g0block << inverse(iOmega_n + u_int/2. + shift * setup['tp'] -
                               0.25*imp_sol.G_iw[name])

        imp_sol.solve(h_int=h_int, **setup['s_params'])

        if mpi.is_master_node():
            with HDFArchive(setup['ofile'].format(**setup)) as last_run:
                last_run['/U{}/it{:03}/G_iw'.format(u_int, loop)] = imp_sol.G_iw
                last_run['/U{}/it{:03}/setup'.format(u_int, loop)] = setup

    return imp_sol.G_iw


def do_setup():
    """Set the solver parameters"""

    parser = argparse.ArgumentParser(description='DMFT loop for CTHYB dimer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sweeps', metavar='MCS', type=float, default=int(1e5),
                        help='Number MonteCarlo Measurement')
    parser.add_argument('-therm', type=int, default=int(5e4),
                        help='Monte Carlo sweeps of thermalization')
    parser.add_argument('-meas', type=int, default=40,
                        help='Number of Updates before measurements')
    parser.add_argument('-Niter', metavar='N', type=int,
                        default=10, help='Number of iterations')
    parser.add_argument('-BETA', metavar='B', type=float,
                        default=200., help='The inverse temperature')
    parser.add_argument('-U', nargs='+', type=float,
                        default=[2.7], help='Local interaction strength')
    parser.add_argument('-tp', default=0.18, type=float,
                        help='The dimerization strength')
    parser.add_argument('-ofile', default='DIMER_PM_B{BETA}_tp{tp}.h5',
                        help='Output file shelve')

    parser.add_argument('-new_seed', type=float, nargs=3, default=False,
                        metavar=('U_src', 'U_target', 'avg_over'),
                        help='Resume DMFT loops from on disk data files')
    setup = vars(parser.parse_args())

    setup.update({'s_params': {'move_double': False,
                               'n_cycles': int(setup['sweeps']),
                               'n_warmup_cycles': setup['therm'],
                               'length_cycle': setup['meas'],
                               'measure_pert_order': False,
                               'random_seed': struct.unpack("I", os.urandom(4))[0]}})

    return setup

if __name__ == "__main__":
    SETUP = do_setup()
    G_iw = None
    for u_loop in SETUP['U']:
        G_iw = dmft_loop(SETUP, u_loop, G_iw)
