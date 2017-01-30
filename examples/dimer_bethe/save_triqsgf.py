# -*- coding: utf-8 -*-
r"""
Saving high temp gf
===================

Using a spectral function create the higher temperature versions
"""
# Created Mon Jan 30 19:17:19 2017
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt

from pytriqs.gf.local import *
from pytriqs.archive import HDFArchive

import dmft.common as gf
import py3qs.dimer_lattice as dlat
from pymaxent.tools import hilbert_trans, differential_weight

tp = 0.3
u_str = 'U1.8'
#u_str = 'U2.01'
#u_str = 'U2.15'

data = np.load('AwmetB25tp0.3{}G_sym.npz'.format(u_str))
Aw = data['Aw']
w = data['w']

# plt.plot(w, Aw)
# plt.show()

###############################################################################
# High temperature seeds
# ----------------------

hot_beta = np.round(1 / np.arange(1 / 25, .2, 1.44e-3), 3)
gfsiw = []
wnli = []
for beta in hot_beta:
    freq = 2 * int(beta * 3)
    wnh = gf.matsubara_freq(beta, freq, 1 - freq)
    wnli.append(wnh)
    print(beta)
    gfsiw.append(hilbert_trans(1j * wnh, w, differential_weight(w), Aw, 0))
    plt.plot(wnh, gfsiw[-1].imag, 'o:')

    # triqs blocks
    gfarr = GfImFreq(indices=[0], beta=beta, n_points=int(beta * 3))
    G_iw = BlockGf(name_list=['asym_dw', 'asym_up', 'sym_dw', 'sym_up'],
                   block_list=(gfarr, gfarr, gfarr, gfarr), make_copies=True)

    for name, gblock in G_iw:
        if 'sym' in name:
            gblock.data[:, 0, 0] = gfsiw[-1]
        else:
            gblock.data[:, 0, 0] = -gfsiw[-1].conj()

    dlat.paramagnetic_hf_clean(G_iw, float(u_str[1:]), tp)

    with HDFArchive('DIMER_PM_met_B{}_tp0.3.h5'.format(beta), 'a') as dest:
        dest['/{}/it000/G_iw'.format(u_str)] = G_iw
