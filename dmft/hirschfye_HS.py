# -*- coding: utf-8 -*-
"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm
"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.linalg import solve
from scipy.linalg.blas import dger
from scipy.interpolate import interp1d

from dmft.common import tau_wn_setup, gw_invfouriertrans, greenF
import hffast


def ising_v(dtau, U, L=32):
    """Initialize the vector of the Hubbard-Stratonovich fields

    .. math:: V = \\Delta \\tau M_l

    where the vector entries :math:`M_l` are normaly distributed according to

    .. math:: P(M_l) \\alpha \\exp(-  \\frac{\\Delta \\tau}{2U} M_l^2)

    Parameters
    ----------
    dtau : float
        time spacing :math:`\\Delta \\tau`
    U : float
        local Coulomb repulsion
    L : integer
        length of the array

    Returns
    -------
    out : single dimension ndarray
    """
    return dtau * np.random.normal(0, np.sqrt(U/dtau), L)