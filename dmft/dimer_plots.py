# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:08:23 2015

@author: oscar
"""

from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq
from pytriqs.plot.mpl_interface import oplot
import dmft.RKKY_dimer as rt
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


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


def plot_gf_loopU(beta, tab, U, filestr, nf,
                  in_box=[0.16, 0.17, 0.20, 0.25]):
    """Loop over all interaction strengths for a given file
    to plot all its iterations"""
    with rt.HDFArchive(filestr.format(tab, beta), 'r') as R:

        f, ax = plt.subplots(1, 2, figsize=(18, 8), sharex=True)
        f.subplots_adjust(hspace=0.2)
        gfin = f.add_axes(in_box)
        gflen = 3*nf
        w_n = gf.matsubara_freq(beta, gflen)
        ru = 'U'+str(U)
        diag_f, offdiag_f = plot_gf_iter(R, ru, gfin, w_n, nf, gflen)

        plt.axhline()
        for freq, (hd, ho) in enumerate(zip(diag_f, offdiag_f)):
            ax[0].plot(hd, 'o-.', label='n='+str(freq+1))
            ax[1].plot(ho, 'o-.', label='n='+str(freq+1))
        ax[1].legend(loc=9, ncol=nf)
        ax[0].set_title('First frequencies of the Matsubara GF, at iteration\n'
                        '@ U/D={} $t_{{ab}}/D={}$ $\\beta D={}$'.format(ru,
                                                                        tab,
                                                                        beta))
        plt.show()
        plt.close()


def plot_acc(filelist):
    """Plot the evolution of the acceptance rate in each DMFT loop
    extracting the status information of the jobs"""
    rawdata = ''
    for fname in filelist:
        with open(fname) as fcontent:
            rawdata += fcontent.read()

    infocols = re.findall('acc\s+([\d\.]+?) nsign (\d)\s+'
                          'B ([\d\.]+) tp ([\d\.]+) U: ([\d\.]+) '
                          'l: (\d+) \w+ ([\d\.]+)', rawdata, flags=re.M)
    infosum = np.array(infocols).astype(np.float)
    table = pd.DataFrame(infosum, columns=['acc', 'sign', 'beta', 'tp', 'U',
                                           'loop', 'dist'])
    tpg = table.groupby('tp')
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
        tp_ax[0].legend(loc=0, ncol=2)
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


def plot_tails(beta, U, tp):
    w_n = gf.matsubara_freq(beta, beta)
    plt.plot(w_n, -1/w_n, '--')
    plt.plot(w_n, -tp/w_n**2, '--')
    plt.plot(w_n, -1/w_n + U**2/4/w_n**3, '--')


def plotGiw(saveblock):
    giw = getGiw(saveblock)
    oplot(giw['A', 'B'], RI='R')
    oplot(giw['A', 'A'], RI='I')
    plot_tails(giw)
    plt.xlim(xmax=10)
    plt.ylim(ymin=-1.2)


def phase_diag(beta):

    fl_dos = []
    w_n = gf.matsubara_freq(beta, 3)
    for tp in np.arange(0.18, 0.3, 0.01):
        filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
        with rt.HDFArchive(filestr, 'r') as results:
            for u in results.keys():
                lastit = results[u].keys()[-1]
                fl_dos.append(gf.fit_gf(w_n, results[u][lastit]['G_iwd'].imag)(0.))
    return np.asarray(fl_dos)


def compare_last_gf(tp, beta, contl=2, ylim=-1):
    """Compartes in a plot the last GF of the HF simulation with the one CTHYB
       run done with this seed"""

    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
    w_n = gf.matsubara_freq(beta, 3)
    f, (gd, go) = plt.subplots(1, 2, figsize=(18, 8))
    with rt.HDFArchive(filestr, 'r') as results:
        for u in results.keys():
            lastit = results[u].keys()[-1]
            gd.oplot(results[u][lastit]['G_iwd'], 'x-', RI='I', label=u)
            go.oplot(results[u][lastit]['G_iwo'], '+-', RI='R', label=u)
            giw = results[u][lastit]['G_iwd']

            fit = gf.fit_gf(w_n, giw[0, 0].imag)
            w = np.arange(0, contl, 0.05)
            gcont = fit(w)
            gd.plot(w, gcont, 'k:')

    gd.set_xlim([0, 4])
    gd.set_ylim([ylim, 0])
    gd.legend(loc=0, prop={'size': 18})
    gd.set_ylabel(r'$\Im m G_{AA}(i\omega_n)$')
    go.set_xlim([0, 4])
    go.legend(loc=0, prop={'size': 18})
    go.set_ylabel(r'$\Re e G_{AB}(i\omega_n)$')
    plt.suptitle('Matsubara GF $t_{{ab}}/D={}$ $\\beta D={}$'.format(tp, beta))


def plot_gf(tp, beta, runsolver, xlim=4, ylim=1.4):

    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
    w_n = gf.matsubara_freq(beta, beta)
    f, (gd, go) = plt.subplots(1, 2, figsize=(18, 8))
    with rt.HDFArchive(filestr, 'r') as results:
        for u in results.keys():
            if runsolver == 'HF':
                lastit = results[u].keys()[-1]
                gd.oplot(results[u][lastit]['G_iwd'], 'x-', RI='I', label=u)
                go.oplot(results[u][lastit]['G_iwo'], '+-', RI='R', label=u)
            if runsolver == 'cthyb':
                lastit = 'cthyb'
                giw = 0.5 * (results[u][lastit]['G_iw']['up'] +
                             results[u][lastit]['G_iw']['down'])
                gd.oplot(giw[0, 0], 'x-', RI='I', label=u)
                go.oplot(giw[0, 1], '+-', RI='R', label=u)

            gd.plot(w_n, -1/w_n + float(u[1:])**2/4/w_n**3, '--')
    gd.set_xlim([0, xlim])
    go.set_xlim([0, xlim])
    gd.set_ylim([-ylim, 0])
    gd.legend(loc=0)
    go.legend(loc=0)
    gd.set_ylabel(r'$\Im m G_{AA}(i\omega_n)$')
    go.set_ylabel(r'$\Re e G_{AB}(i\omega_n)$')
    plt.suptitle('Matsubara GF $t_{{ab}}/D={}$ $\\beta D={}$'.format(tp, beta))


def diag_sys(tp, beta):
    rot = np.matrix([[-1, 1], [1, 1]])/np.sqrt(2)
    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
    f, (gd, go) = plt.subplots(1, 2, figsize=(18, 8))
    with rt.HDFArchive(filestr, 'r') as results:
        for u in results.keys():
            lastit = results[u].keys()[-1]
            g_iw = rot*getGiw(results[u][lastit])*rot
            gd.oplot(g_iw['A', 'A'], 'x-', label=u)
            go.oplot(g_iw['B', 'B'], '+-', label=u)

    gd.set_xlim([0, 4])
    gd.legend(loc=0, prop={'size': 18})
    gd.set_ylabel(r'$\Im m G_{1}(i\omega_n)$')
    go.set_xlim([0, 4])
    go.legend(loc=0, prop={'size': 18})
    go.set_ylabel(r'$\Re e G_{2}(i\omega_n)$')
    plt.suptitle('Matsubara GF $t_{{ab}}/D={}$ $\\beta D={}$'.format(tp, beta))


def spectral(tp, U, beta, pade_fit_pts):
    rot = np.matrix([[-1, 1], [1, 1]])/np.sqrt(2)
    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
    f, (gl, gd) = plt.subplots(1, 2, figsize=(18, 8))
    with rt.HDFArchive(filestr, 'r') as results:
        u = 'U'+str(U)
        lastit = results[u].keys()[-1]
        g_iw = getGiw(results[u][lastit])
        greal = GfReFreq(indices=[0, 1], window=(-3.5, 3.5), n_points=500)
        greal.set_from_pade(g_iw, pade_fit_pts, 0.)
        gl.oplot(greal[0, 0], RI='S', label='out')
        gl.set_title('On site GF, fit pts'+str(pade_fit_pts))
        gl.set_ylim([0, 0.6])

        rgiw = rot*g_iw*rot
        greal.set_from_pade(rgiw, pade_fit_pts, 0.)
        gd.oplot(greal[0, 0], RI='S', label='bond')
        gd.oplot(greal[1, 1], RI='S', label='anti-bond')
        gd.set_title('Diagonal GF')

        gl.oplot(0.5*(greal[0, 0] + greal[1, 1]), '--', RI='S', label='d avg')

    plt.show()
    plt.close()


def plotginta():
    U, tp, beta = 2.8, 0.18, 60.
    filestr = 'disk/metf_HF_Ul_tp{}_B{}.h5'.format(tp, beta)
    with rt.HDFArchive(filestr, 'r') as results:
        u = 'U'+str(U)
        lastit = results[u].keys()[-1]
        oplot(results[u][lastit]['G_iwd'], 'x-', RI='I', label=u)
        w_n = gf.matsubara_freq(beta, beta)
        plt.plot(w_n, -1/w_n, '--')
        plt.plot(w_n, -1/w_n + float(u[1:])**2/4/w_n**3, '--')
    plt.xlim([0, 6])
    plt.ylim([-.6, 0])
