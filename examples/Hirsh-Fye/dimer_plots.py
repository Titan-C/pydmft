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


def plot_gf_iter(R, ru, gfin, w_n, nf, gflen):
    diag_f = []
    offdiag_f = []

    for u_iter in R[ru].keys():
        diag_f.append(R[ru][u_iter]['G_iwd'].data[:nf, 0, 0].imag)
        offdiag_f.append(R[ru][u_iter]['G_iwo'].data[:nf, 0, 0].real)
        gfin.plot(w_n, R[ru][u_iter]['G_iwd'].data[:gflen, 0, 0].imag, 'bs:')
        gfin.plot(w_n, R[ru][u_iter]['G_iwo'].data[:gflen, 0, 0].real, 'gs:')
    diag_f = np.asarray(diag_f).T
    offdiag_f = np.asarray(offdiag_f).T
#    gfin.plot(w_n[2*nf:], -1/w_n[2*nf:])  # tails
#    gfin.plot(w_n[2*nf:], -tab/w_n[2*nf:]**2)
    gfin.set_xticks(gfin.get_xlim())
    gfin.set_yticks(gfin.get_ylim())

    return diag_f, offdiag_f

def plot_gf_loopU(beta, tab, U, filestr, nf):
    R = rt.HDFArchive(filestr.format(tab, beta), 'r')

    f, ax = plt.subplots(1, 2, figsize=(18, 8), sharex=True)
    f.subplots_adjust(hspace=0.2)
    gfin = f.add_axes([0.16, 0.17, 0.20, 0.25])
    gflen = 3*nf
    w_n = gf.matsubara_freq(beta, gflen)
    ru = 'U'+str(U)
    diag_f, offdiag_f = plot_gf_iter(R, ru, gfin, w_n, nf, gflen)

    plt.axhline()
    for freq, (hd, ho) in enumerate(zip(diag_f, offdiag_f)):
        ax[0].plot(hd, 'o-.', label='n='+str(freq+1))
        ax[1].plot(ho, 'o-.', label='n='+str(freq+1))
    ax[1].legend(loc=3, prop={'size':18})
    plt.suptitle('First frequencies of the Matsubara GF, at iteration\
    @ U/D={} $t_{{ab}}/D={}$ $\\beta D={}$'.format(ru, tab, beta))
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

def getGiw(saveblock):
    gd = saveblock['G_iwd']
    nf = len(gd.mesh)
    beta = gd.beta
    g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=nf)
    rt.load_gf(g_iw, gd, saveblock['G_iwo'])
    return g_iw

def plot_tails(giw):
    w_n = gf.matsubara_freq(giw.beta, len(giw.mesh))
    plt.plot(w_n, -1/w_n, '--')
    plt.plot(w_n, -0.23/w_n**2, '--')

def plotGiw(saveblock):
    giw=getGiw(saveblock)
    oplot(giw['A','B'], RI='R')
    oplot(giw['A','A'], RI='I')
    plot_tails(giw)
    plt.xlim(xmax=10)
    plt.ylim(ymin=-1.2)
