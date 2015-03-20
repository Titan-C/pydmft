# -*- coding: utf-8 -*-
"""
Helper file to deal with HDF5 files generated with the ALPS Library
"""

from dmft import ipt_imag
from dmft.common import matsubara_freq, greenF, gw_invfouriertrans
import numpy as np
from pyalps.hdf5 import archive


def save_pm_delta_tau(parms, gtau):
    """Saves to file and returns the imaginary time hybridization function
    enforcing paramagnetism"""
    save_delta = archive(parms["DELTA"], 'w')
    delta = parms['t']**2 * gtau
    delta[delta>-1e-4] = -1e-4

    save_delta['/Delta_0'] = delta
    save_delta['/Delta_1'] = delta
    del save_delta

    return delta


def recover_measurement(parms, measure):
    """Recovers a specific measurement from the output file"""
    iteration = archive(parms['BASENAME'] + '.out.h5', 'r')
    data = []
    for i in range(parms['N_ORBITALS']):
        data.append(iteration[measure+'/{}/mean/value'.format(i)])
    del iteration
    return np.asarray(data)


def save_iter_step(parms, iter_count, measure, data):
    """Saves the measurement results to log DMFT iterations"""
    save = archive(parms['BASENAME']+'steps.h5', 'w')
    for i, data_vector in enumerate(data):
        save['iter_{:0>2}/{}/{}/'.format(iter_count, measure, i)] = data_vector
    del save


def start_delta(parms):
    """Provides a starting guess for the hybridization function given the
    cthyb impurity solvers parameters. Guess is based on the IPT solution"""

    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    giw = greenF(iwn, mu=0., D=2*parms['t'])
    giw, siw = ipt_imag.dmft_loop(30, parms['U'], parms['t'], giw, iwn, tau)
    gtau = gw_invfouriertrans(giw[-1], tau, iwn)

    return save_pm_delta_tau(parms, gtau)
