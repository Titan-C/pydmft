# -*- coding: utf-8 -*-
r"""
The Dimer phase diagram on temperature
======================================

Collect data on double occupation and plot the phase diagram
"""
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import argparse
import re
from glob import glob
import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

###############################################################################
# Dimer :math:`t_\perp/D=0.3` Phase diagram
# -----------------------------------------

data = np.load('met_tp0.3_UTd.npz', encoding='bytes')
md_list, mtemp_list, mu_list = data['docc'], data['temp'], data['u_int']

data = np.load('ins_tp0.3_UTd.npz', encoding='bytes')
id_list, itemp_list, iu_list = data['docc'], data['temp'], data['u_int']

# Join double occupation results
x = np.concatenate(list(mu_list) + list(iu_list))
y = np.concatenate(list(mtemp_list) + list(itemp_list))
z = np.concatenate(list(md_list) + list(id_list))
# define plot grid.
xi = np.linspace(1.7, 2.5, 150)
yi = np.linspace(0, 0.04, 150)
# Grid the data
zi = griddata(x, y, z, xi, yi, interp='nn')
CS = plt.contourf(xi, yi, zi, 15, cmap=plt.get_cmap('viridis'))
plt.colorbar()  # draw colorbar

# plot data points.
boundaries = np.array([(1.91, 1e5), (1.91, 300.), (1.91, 200.), (1.93, 100.),
                       (1.99, 64.), (2.115, 44.23), (2.145, 41.56),
                       (2.18, 40.), (2.18, 64.), (2.18, 100.), (2.19, 200.),
                       (2.205, 300.), (2.24, 1e5)]).T
DH0 = np.array([(2.05, 1e5), (2.05, 300.), (2.05, 200.), (2.05, 100.),
                (2.05, 64.), (2.07, 50.732), (2.12, 44.23), (2.18, 40.)]).T

plt.plot(DH0[0], 1 / DH0[1], 'rx-', lw=3)
plt.fill(boundaries[0], 1 / boundaries[1], 'k+-', alpha=0.5, lw=4)

plt.scatter(np.concatenate(mu_list), np.concatenate(mtemp_list),
            c=np.concatenate(md_list), s=70, vmin=0.036, vmax=0.12,
            cmap=plt.get_cmap('viridis'), marker='o', edgecolor='k')

plt.scatter(np.concatenate(iu_list), np.concatenate(itemp_list),
            c=np.concatenate(id_list), s=30, vmin=0.036, vmax=0.12,
            cmap=plt.get_cmap('viridis'), marker='o', edgecolor='k')

plt.xlim([1.7, 2.5])
plt.ylim([0, 0.04])

plt.xlabel(r'$U/D$')
plt.ylabel(r'$T/D$')
