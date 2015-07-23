# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:08:23 2015

@author: oscar
"""
from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.gf.local import GfImFreq, iOmega_n, TailGf, inverse
from pytriqs.gf.local import GfImTime, InverseFourier, Fourier
from pytriqs.plot.mpl_interface import oplot
import matplotlib.pyplot as plt
import numpy as np
import dmft.common as gf
import dmft.RKKY_dimer as rt


def plot_gf_loops(tab, beta, filestr, nf):
    R = rt.HDFArchive(filestr.format(tab, beta), 'r')
    gflen = 3*nf
    w_n = gf.matsubara_freq(beta, gflen)
    for ru in R.keys():
        diag_f = []
        offdiag_f = []
        f, ax = plt.subplots(1, 2, figsize=(18, 8), sharex=True)
        f.subplots_adjust(hspace=0.2)
        gfin = f.add_axes([0.16, 0.17, 0.20, 0.25])
        for u_iter in R[ru].keys():
            if 'it' in u_iter:
                diag_f.append(R[ru][u_iter]['G_iwd'].data[:nf, 0, 0].imag)
                offdiag_f.append(R[ru][u_iter]['G_iwo'].data[:nf, 0, 0].real)
                gfin.plot(w_n, R[ru][u_iter]['G_iwd'].data[:gflen, 0, 0].imag, 'bs:')
                gfin.plot(w_n, R[ru][u_iter]['G_iwo'].data[:gflen, 0, 0].real, 'gs:')

        diag_f = np.asarray(diag_f).T
        offdiag_f = np.asarray(offdiag_f).T
        gfin.plot(w_n[2*nf:], -1/w_n[2*nf:])
        gfin.plot(w_n[2*nf:], -tab/w_n[2*nf:]**2)
        gfin.set_xticks(gfin.get_xlim())
        gfin.set_yticks(gfin.get_ylim())
        plt.axhline()
        for freq, (hd, ho) in enumerate(zip(diag_f, offdiag_f)):
            ax[0].plot(hd,'o-.', label='n='+str(freq+1))
            ax[1].plot(ho,'o-.', label='n='+str(freq+1))
        ax[1].legend(loc=3,prop={'size':18})
        plt.suptitle('First frequencies of the Matsubara GF, at iteration @ U/D={} $t_{{ab}}/D={}$ $\\beta D={}$'.format(ru, tab, beta))
        #show()
        #close()
    del R


def get_selfE(G_iwd, G_iwo):
    nf = len(G_iwd.mesh)
    beta = G_iwd.beta
    g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=nf)
    rt.load_gf(g_iw, G_iwd, G_iwo)
    gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, 0, 0.2)
    sigma = g_iw.copy()
    sigma << gmix - 0.25*g_iw - inverse(g_iw)
    return sigma