# -*- coding: utf-8 -*-
r"""
===========================================
Evolution of DOS as function of temperature
===========================================

Using a real frequency solver in the IPT scheme the Density of states
is tracked through the first orders transition.
"""
# Created Tue Jun 14 15:44:38 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt
plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22})
import numpy as np

import dmft.common as gf
import dmft.RKKY_dimer as rt
import dmft.ipt_real as ipt
from slaveparticles.quantum.operators import fermi_dist
import slaveparticles.quantum.dos as dos
import scipy.signal as signal
from scipy.integrate import trapz


def loop_beta(u_int, tp, betarange, seed='mott gap'):
    """Solves IPT dimer and return Im Sigma_AA, Re Simga_AB

    returns list len(betarange) x 2 Sigma arrays
"""

    s = []
    g = []
    w = np.linspace(-6, 6, 2**13)
    dw = w[1] - w[0]
    gss = gf.semi_circle_hiltrans(w + 5e-3j - tp - 1)
    gsa = gf.semi_circle_hiltrans(w + 5e-3j + tp + 1)
    for beta in betarange:
        print('U: ', u_int, 'tp: ', tp, 'Beta', beta)
        nfp = dos.fermi_dist(w, beta)
        (gss, gsa), (ss, sa) = ipt.dimer_dmft(u_int, tp, nfp, w, dw, gss, gss)
        g.append((gss, gsa))
        s.append((ss, sa))

    return np.array(g), np.array(s)

U = 2.5
tp = 0.3

temp = np.linspace(0.0165, 0.0285, 50)
betarange = np.concatenate((np.array([100., 90., 80., 70., ]), 1 / temp))


try:
    storage = np.load('dimer_Tevol_U2.5tp0.3')
    temp = storage['temp']
    betarange = storage['betarange']
    gwi = storage['giw']
    swi = storage['siw']
except FileNotFoundError:
    gwi, swi = loop_beta(U, tp, betarange)
    np.savez('dimer_Tevol_U2.5tp0.3.npz', temp=temp,
             betarange=betarange, gwi=gwi, swi=swi)


plt.figure()
w = np.linspace(-6, 6, 2**13)
for (gss, gsa), beta in zip(gwi, betarange):
    gloc = .5 * (gss + gsa)
    plt.plot(w, 100 / beta - gloc.imag / np.pi)

plt.yticks(100 / betarange[[0, 1, 2, 3, 4, - 1]],
           np.around(1 / betarange, 3)[[0, 1, 2, 3, 4, - 1]])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega)$')
plt.xlim([-3, 3])
plt.savefig('dimer_sig_Tevo_U2.5tp.3_iptre_yshift.pdf')
###############################################################################

plt.figure()
for (gss, gsa), beta in zip(gwi, betarange):
    gloc = .5 * (gss + gsa)
    plt.plot(w, - gloc.imag / np.pi)

plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega)$')
plt.xlim([-3, 3])
plt.savefig('dimer_dos_Tevo_U2.5tp.3_iptre_ontop.pdf')

###############################################################################
plt.xlim([-.3, .3])
plt.ylim([0, 0.05])
plt.savefig('dimer_dos_Tevo_U2.5tp.3_iptre_gapzoom.pdf')

###############################################################################
plt.xlim([.4, 1.2])
plt.ylim([0.2, 0.4])
plt.savefig('dimer_dos_Tevo_U2.5tp.3_iptre_qp_zoom.pdf')
# plt.close('all')

###############################################################################


def bubble(A1, A2, nf):
    # sig = int dw A(w)*A(w+w')*(nf(w)-nf(w+w'))
    #     = int dw A^+(w)*A(w+w')-A(w)*A^+(w+w')
    return signal.fftconvolve((A1 * nf)[::-1], A2, mode='same') - \
        signal.fftconvolve(A1[::-1], A2 * nf, mode='same')


def pol_bubble_conv(lat_A1, lat_A2, nf, w, dosde):
    dw = w[1] - w[0]
    lat_sig = np.array([bubble(A1, A2, nf) for A1, A2 in zip(lat_A1, lat_A2)])
    resig = (dosde * lat_sig).sum(axis=0) * dw / (w - dw / 2)
    center = int(len(w) / 2)
    resig[center] = (resig[center - 1] + resig[center + 1]) / 2
    return resig


def optical_cond(ss, sa, tp, w, beta):
    E = np.linspace(-1, 1, 61)
    dos = np.exp(-2 * E**2) / np.sqrt(np.pi / 2)
    de = E[1] - E[0]
    dosde = (dos * de).reshape(-1, 1)
    nf = fermi_dist(w, beta)

    lat_Aa = (-1 / np.add.outer(-E, w + tp - sa)).imag / np.pi
    lat_As = (-1 / np.add.outer(-E, w - tp - ss)).imag / np.pi

    a = pol_bubble_conv(lat_Aa, lat_Aa, nf, w, dosde)
    b = pol_bubble_conv(lat_As, lat_As, nf, w, dosde)
    c = pol_bubble_conv(lat_Aa, lat_As, nf, w, dosde)
    d = pol_bubble_conv(lat_As, lat_Aa, nf, w, dosde)

    return a, b, c, d
resi = []
for (ss, sa), beta in zip(swi, betarange):
    resi.append(np.sum(optical_cond(ss, sa, tp, w, beta), axis=0))

# Reference insulator data from http://dx.doi.org/10.1126/science.1150124
s_ins = np.loadtxt(
    '/home/oscar/Dropbox/org/phd/dimer_lattice/optical conductivity/ins_340.csv', comments='x', delimiter=',').T
s_exp_weight = trapz(s_ins[1], x=s_ins[0] / 8056.54)
rdw = (w > 0) * (w < s_ins[0][-1] / 8056.54)
sim_weight = trapz(resi[0][rdw], x=w[rdw])

plt.figure()
for res, beta in zip(resi, betarange):
    plt.plot(w, s_exp_weight / sim_weight * res)

plt.plot(w, s_exp_weight / sim_weight * res, lw=2)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\Re \sigma(\omega)$')
plt.ylim([0, 6000])
plt.xlim([0, 3])
plt.show()
plt.savefig('dimer_resigma_Tevo_U2.5tp.3_iptre.pdf')

###############################################################################
plt.xlim([0, 0.6])
plt.ylim([0., 0.8])
plt.savefig('dimer_resigma_Tevo_U2.5tp.3_iptre_gapzoom.pdf')

plt.figure()
dw = w[1] - w[0]
epsilon = []
for res, beta in zip(resi, betarange):
    sig = signal.hilbert(res, len(res) * 4)[:len(res)]
    ep = 1 + 4 * np.pi * 1j * sig / \
        (w - dw / 2) * s_exp_weight / sim_weight / 8056.54
    epsilon.append(ep)
    plt.plot(w, ep.real, '-', w, ep.imag, ':')
plt.axvline(w[4174])
plt.axvline(w[4247])
plt.ylim([-40, 40])
plt.xlim([0.01, .6])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\varepsilon$')
plt.savefig('dimer_epsilon_Tevo_U2.5tp.3_iptre.pdf')

###############################################################################

plt.figure()
epr = []
epp = []
for ep in epsilon:
    epr.append(ep[4174])
    epp.append(ep[4247])
epr = np.array(epr)
epp = np.array(epp)
plt.plot(1 / betarange, epr.real, 'g', label=r'$\omega=0.115$')
plt.plot(1 / betarange, epr.imag, "g--")
plt.plot(1 / betarange, epp.real, 'r', label=r'$\omega=0.222$')
plt.plot(1 / betarange, epp.imag, "r--")
plt.xlabel('T')
plt.ylabel(r'$\varepsilon$')
plt.legend(loc=0)
plt.savefig('dimer_epsilon_Tevo_U2.5tp.3_iptre_fixfreq.pdf')
###############################################################################


def aal_ef(eps_sample, t, e_tip=-1300 + 960j, a=20):
    beta = (eps_sample - 1) / (eps_sample + 2)
    alpha = 4 * np.pi * a ** 3 * (e_tip - 1) / (e_tip + 2)
    return alpha * (1 + beta) / (1 - alpha * beta / (16 * np.pi * (3 * a + 2 * a * np.cos(t))**3))


t = np.linspace(-np.pi, np.pi, 600)

plt.figure()
sam = []
for ep in epr:
    am = aal_ef(ep, t)
    sam.append(abs(trapz(am * np.cos(2 * t), t)))

sap = []
for ep in epp:
    am = aal_ef(ep, t, -474.41 + 248.61j)
    sap.append(abs(trapz(am * np.cos(2 * t), t)))

# This normalization because this is an intensity measure so just to have
# readable numbers
sam = np.array(sam) / 1e4
sap = np.array(sap) / 1e4
plt.plot(1 / betarange, sam, 'g', label=r'$\omega=0.115$')
plt.plot(1 / betarange, sap, 'r', label=r'$\omega=0.222$')
plt.xlabel('T')
plt.ylabel('2$^{nd}$ harmonic amplitude')
plt.legend(loc=0)
plt.savefig('dimer_s2_Tevo_U2.5tp.3_iptre_fixfreq.pdf')
