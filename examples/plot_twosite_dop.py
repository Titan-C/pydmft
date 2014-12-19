# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""
from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from dmft.twosite_dop import dmft_loop_dop


def doping_config(res, name):
    fig, axes = plt.subplots(3, sharex=True)
    axes[-1].set_xlabel('$<N>_{imp}$')
    fill = res[:, 0]
    axes[0].set_xlim([0, 1])
    e_c = [sim.e_c for sim in res[:, 1]]
    V = [sim.hyb_V() for sim in res[:, 1]]
    mu = [sim.mu for sim in res[:, 1]]
    for feat, ax, lab in zip([e_c, V, mu], axes, ['$\\epsilon_c$', 'V', '$\\mu$']):
        ax.plot(fill, feat, label=lab)
        ax.set_ylabel(lab)

    fig.savefig(name+'_bathparam.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)


def plot_doping_param(axis='real', beta=1e3, u_int=[4.]):
    for u in u_int:
        out_file = axis+'_dop_b{}_U{}'.format(beta, u)
        try:
            res = np.load(out_file+'.npy')
        except IOError:
            res = dmft_loop_dop(u)
            np.save(out_file, res)

        doping_config(res, out_file)

if __name__ == "gallery":
    plot_doping_param(u_int=[2., 4., 5.85, 6., 8., 10., 100.])
