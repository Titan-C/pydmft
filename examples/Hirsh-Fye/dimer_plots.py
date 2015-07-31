# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:08:23 2015

@author: oscar
"""

from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, TailGf, inverse
from pytriqs.gf.local import GfImTime, InverseFourier, Fourier
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.plot.mpl_interface import oplot
import dmft.RKKY_dimer as rt
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
plt.matplotlib.rcParams.update({'font.size': 22})

def plot_gf_iter(R, ru, gfin, w_n, nf, gflen):
    """Plot all Hirsch - Fye iterations of a given file at
    the specified interaction strength"""
    diag_f = []
    offdiag_f = []

    for u_iter in R[ru].keys():
        if u_iter == 'cthyb':
            continue
        diag_f.append(R[ru][u_iter]['G_iwd'].data[:nf, 0, 0].imag)
        offdiag_f.append(R[ru][u_iter]['G_iwo'].data[:nf, 0, 0].real)
        gfin.oplot(R[ru][u_iter]['G_iwd'], 'bs:', RI='I')
        gfin.oplot(R[ru][u_iter]['G_iwo'], 'gs:', RI='R')
    diag_f = np.asarray(diag_f).T
    offdiag_f = np.asarray(offdiag_f).T
#    gfin.plot(w_n[2*nf:], -1/w_n[2*nf:])  # tails
#    gfin.plot(w_n[2*nf:], -tab/w_n[2*nf:]**2)
    gfin.set_xlim([0, 5])
    gfin.set_xticks(gfin.get_xlim())
    gfin.set_yticks(gfin.get_ylim())
    gfin.legend_.remove()

    return diag_f, offdiag_f

def plot_gf_loopU(beta, tab, U, filestr, nf):
    """Loop over all interaction strengths for a given file
    to plot all its iterations"""
    with rt.HDFArchive(filestr.format(tab, beta), 'r') as R:

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
        plt.show()
        plt.close()

def plot_acc(filelist):
    """Plot the evolution of the acceptance rate in each DMFT loop
    extracting the status information of the jobs"""
    rawdata=''
    for fname in filelist:
        with open(fname) as fcontent:
            rawdata += fcontent.read()

    infocols=re.findall('acc\s+([\d\.]+?) nsign (\d)\s+B ([\d\.]+) tp ([\d\.]+) U: ([\d\.]+) l: (\d+) \w+ ([\d\.]+)', rawdata, flags=re.M)
    infosum=np.array(infocols).astype(np.float)
    table=pd.DataFrame(infosum,columns=['acc', 'sign', 'beta', 'tp', 'U', 'loop', 'dist'])
    tpg=table.groupby('tp')
    for tp_key, group in tpg:
        tpg_ug = group.groupby('U')
        f, tp_ax = plt.subplots(1, 2, figsize=(18, 8))
        for U_key, ugroup in tpg_ug:
            ugroup.plot(x='loop', y='acc', ax=tp_ax[0], marker='o',
                        label='U='+str(U_key),
                        title='Acceptance rate @ $t\\perp=$'+str(tp_key))
            ugroup.plot(x='loop', y='dist', ax=tp_ax[1], marker='o',
                        label='U='+str(U_key), logy=True, legend=False,
                        title='Convergence @ $t\\perp=$'+str(tp_key))
        tp_ax[0].legend(loc=0, prop={'size':18})
        tp_ax[1].axhline(y=4e-3, ls=':')
        plt.show()
        plt.close()

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

def phase_diag(beta):

    fl_dos = []
    for tp in np.arange(0.18, 0.3, 0.01):
        w_n = gf.matsubara_freq(beta, 5)
        filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
        with rt.HDFArchive(filestr, 'r') as results:
            for u in results.keys():
                lastit = results[u].keys()[-1]
                fl_dos.append(rt.fit_dos(w_n, results[u][lastit]['G_iwd'])(0.))
    return np.asarray(fl_dos)

def plot_gf(tp, beta):

    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
    f, (gd, go) = plt.subplots(1, 2, figsize=(18, 8))
    with rt.HDFArchive(filestr, 'r') as results:
        for u in results.keys():
            lastit = results[u].keys()[-1]
            gd.oplot(results[u][lastit]['G_iwd'], RI='I', label=u)
            go.oplot(results[u][lastit]['G_iwo'], RI='R', label=u)

    gd.set_xlim([0, 4])
    gd.legend(loc=0, prop={'size': 18})
    go.set_xlim([0, 4])
    go.legend(loc=0, prop={'size': 18})
