#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dimer in Bethe lattice
======================

"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import struct
from itertools import combinations
from math import sqrt

import pytriqs.utility.mpi as mpi
import numpy as np
from pytriqs.applications.impurity_solvers.cthyb import Solver, trace_rho_op
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import SemiCircular, iOmega_n, inverse
from pytriqs.operators import c, dagger

import dmft.plot.triqs_dimer as tdimer
import dmft.RKKY_dimer as rt
import dmft.common as gf


def prepare_interaction(u_int):
    """Build the local interaction term of the dimer

    using the symmetric anti-symmetric basis"""

    aup = 1 / sqrt(2) * (-c('asym_up', 0) + c('sym_up', 0))
    adw = 1 / sqrt(2) * (-c('asym_dw', 0) + c('sym_dw', 0))

    bup = 1 / sqrt(2) * (c('asym_up', 0) + c('sym_up', 0))
    bdw = 1 / sqrt(2) * (c('asym_dw', 0) + c('sym_dw', 0))

    h_int = u_int * (dagger(aup) * aup * dagger(adw) * adw +
                     dagger(bup) * bup * dagger(bdw) * bdw)
    return h_int, [aup, bup, adw, bdw]


def density_occup(solver_core, operators):

    nn_avg = []
    for cop in operators:
        nn_avg.append(trace_rho_op(solver_core.density_matrix,
                                   dagger(cop) * cop,
                                   solver_core.h_loc_diagonalization))
    return np.array(nn_avg)


def density_correlators(solver_core, operators):
    flavor_pairs = list(combinations(range(4), 2))

    nn_avg = []
    for i, j in flavor_pairs:
        n_i_x_n_j = dagger(operators[i]) * operators[i] * \
            dagger(operators[j]) * operators[j]
        nn_avg.append(trace_rho_op(solver_core.density_matrix,
                                   n_i_x_n_j,
                                   solver_core.h_loc_diagonalization))
    return np.array(nn_avg)


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
        dest_group = '/{}/it{:03}/'.format(dest_U, dest_count)

        outp[dest_group + 'G_iw'] = giw

    print(setup['new_seed'])


def dmft_loop(setup, u_int, G_iw):
    """Starts impurity solver with DMFT paramagnetic self-consistency"""
    setup['simt'] = 'AFM' if setup['AFM'] else 'PM'

    if setup['new_seed']:
        set_new_seed(setup)
        return

    imp_sol = Solver(beta=setup['BETA'],
                     gf_struct={'asym_up': [0], 'sym_up': [0],
                                'asym_dw': [0], 'sym_dw': [0]},
                     n_iw=int(3 * setup['BETA']), n_tau=int(100 * setup['BETA']))
    work_pt = len(imp_sol.G_iw.mesh) / 2
    h_int, operators = prepare_interaction(u_int)

    src_U = 'U' + str(u_int)

    try:
        with HDFArchive(setup['ofile'].format(**setup), 'r') as outp:
            g_iw_seed = tdimer.get_giw(outp[src_U], slice(-1, None))
            tdimer.paramagnetic_hf_clean(g_iw_seed, u_int, setup['tp'])
            last_loop = len(outp[src_U])
            src_pt = len(g_iw_seed.mesh) / 2
            try:
                for name, gblock in imp_sol.G_iw:
                    gblock.data[:] = g_iw_seed[name].data[
                        src_pt - work_pt:src_pt + work_pt]
            except IndexError:
                import itertools
                spin = ['up', 'dw']
                newn = (''.join(a)
                        for a in itertools.product(['asym_', 'sym_'], spin))
                oldn = (''.join(a)
                        for a in itertools.product(['high_', 'low_'], spin))
                for name_n, name_o in zip(newn, oldn):
                    imp_sol.G_iw[name_n].data[:] = g_iw_seed[
                        name_o].data[src_pt - work_pt:src_pt + work_pt]
    except (KeyError, IOError):
        last_loop = 0
        freqs = int(3 * setup['BETA'])
        tau, w_n = gf.tau_wn_setup(dict(BETA=setup['BETA'], N_MATSUBARA=freqs))
        giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
        giw_d, giw_o, loops = rt.ipt_dmft_loop(setup['BETA'], u_int, setup[
                                               'tp'], giw_d, giw_o, tau, w_n)
        gss = giw_d + giw_o
        gsa = giw_d - giw_o
        gss = np.concatenate((gss[::-1], gss))
        gsa = np.concatenate((gsa[::-1], gsa))
        for name, gblock in imp_sol.G_iw:
            gblock.data[:] = gsa if 'asym' in name else gss

    if G_iw is not None:
        imp_sol.G_iw << G_iw

    for loop in range(last_loop, last_loop + setup['Niter']):
        if mpi.is_master_node():
            print('\n\n in loop \n', '=' * 40, loop)

        if not setup['AFM']:
            tdimer.paramagnetic_hf_clean(imp_sol.G_iw, u_int, setup['tp'])

        for name, g0block in imp_sol.G0_iw:
            shift = 1. if 'asym' in name else -1
            if setup['AFM']:
                if 'up' in name:
                    name_n = name.replace('up', 'dw')
                else:
                    name_n = name.replace('dw', 'up')
            else:
                name_n = name
            g0block << inverse(iOmega_n + u_int / 2. + shift * setup['tp'] -
                               0.25 * imp_sol.G_iw[name_n])

        imp_sol.solve(h_int=h_int, **setup['s_params'])

        if mpi.is_master_node():
            with HDFArchive(setup['ofile'].format(**setup)) as last_run:
                last_run[
                    '/U{}/it{:03}/G_iw'.format(u_int, loop)] = imp_sol.G_iw
                last_run['/U{}/it{:03}/setup'.format(u_int, loop)] = setup
                last_run[
                    '/U{}/it{:03}/density'.format(u_int, loop)] = density_correlators(imp_sol, operators)
                last_run[
                    '/U{}/it{:03}/occup'.format(u_int, loop)] = density_occup(imp_sol, operators)

                if setup['save_gtau']:
                    last_run[
                        '/U{}/it{:03}/G_tau'.format(u_int, loop)] = imp_sol.G_tau

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
    parser.add_argument('-ofile', default='DIMER_{simt}_B{BETA}_tp{tp}.h5',
                        help='Output file shelve')

    parser.add_argument('-afm', '--AFM', action='store_true',
                        help='Use the self-consistency for Antiferromagnetism')
    parser.add_argument('-sg', '--save_gtau', action='store_true',
                        help='Saves Gtau in file')
    parser.add_argument('-dm', '--double_moves', action='store_true',
                        help='Enables double moves during sampling')
    parser.add_argument('-pert', action='store_true',
                        help='Saves the perturbation order histogram')
    parser.add_argument('-new_seed', type=float, nargs=3, default=False,
                        metavar=('U_src', 'U_target', 'avg_over'),
                        help='Resume DMFT loops from on disk data files')
    setup = vars(parser.parse_args())

    setup.update({'s_params': {'move_double': setup['double_moves'],
                               'n_cycles': int(setup['sweeps']),
                               'n_warmup_cycles': setup['therm'],
                               'length_cycle': setup['meas'],
                               'measure_pert_order': setup['pert'],
                               'use_norm_as_weight': True,
                               'measure_density_matrix': True,
                               'random_seed': struct.unpack("I", os.urandom(4))[0]}})

    return setup

if __name__ == "__main__":
    SETUP = do_setup()
    G_iw = None
    for u_loop in SETUP['U']:
        G_iw = dmft_loop(SETUP, u_loop, G_iw)
