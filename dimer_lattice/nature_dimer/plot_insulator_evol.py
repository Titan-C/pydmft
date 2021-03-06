"""
Show the evolution of spectral Function and Self-energy in the insulator
========================================================================

Study the change of increasing dimerization to tame the Mott insulator into
a Peierls insulator. Then from the Peierls insulator increase correlation
to recover Mottness.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

import dmft.common as gf
import dmft.ipt_imag as ipt
import dmft.dimer as dimer

# When True, automatically adjust subplot
plt.rcParams['figure.autolayout'] = False
plt.rcParams["axes.grid"] = False


def ipt_u_tp(u_int, tp, beta, seed='ins'):

    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=2**12))
    giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)

    if 'ins' in seed:
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_d, giw_o, loops = dimer.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-9)
    g0iw_d, g0iw_o = dimer.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    return giw_d, giw_o, siw_d, siw_o, w_n


def ipt_co_s(u_int, tp, BETA, seed, w):
    giw_d, giw_o, siw_d, siw_o, w_n = ipt_u_tp(u_int, tp, BETA, seed)

    w_set = list(np.arange(0, 20, 1))
    w_set = w_set + list(np.arange(20, 120, 2))
    w_set = w_set + list(np.arange(120, 512, 8))
    sa = gf.pade_continuation(
        1j * siw_d.imag - siw_o.real, w_n, w + 0.0005j, w_set)  # A-bond

    sa = sa.real - 1j * np.abs(sa.imag)
    return sa


def low_en_qp(sa):
    glp = np.array([0.])
    sigtck = splrep(w, sa.real, s=0)
    sig_0 = splev(glp, sigtck, der=0)[0]
    dw_sig0 = splev(glp, sigtck, der=1)[0]
    quas_z = 1 / (1 - dw_sig0)
    return quas_z, sig_0, dw_sig0


def plot_spectral(omega, tp, sa, axes, ylim):

    (axgf, axsg) = axes
    quas_z, sig_0, dw_sig0 = low_en_qp(sa)
    tpp = (tp - sig_0) * quas_z
    gsa = gf.semi_circle_hiltrans(w + tp - sa)
    axgf.plot(omega, -gsa.imag / np.pi, 'C0')
    llg = gf.semi_circle_hiltrans(omega + 1e-8j + tpp, quas_z) * quas_z
    axgf.plot(omega, -llg.imag / np.pi,  "C3--", lw=2)
    axgf.tick_params(left='on', right='on')

    # plt.plot(omega, gst.real)
    axsg.plot(omega, sa.real,  'C4', label=r'$\Re e$')
    axsg.plot(omega, sa.imag,  'C2-.', label=r'$\Im m$')
    axsg.legend(loc=2, ncol=2)
    axsg.set_ylim(*ylim)
    axsg.tick_params(left='on', right='on')
    axsg.axhline(0, color='k', lw=0.2)


###############################################################################
# Change from Mott insulator to Peierls by increasing dimerization
# ----------------------------------------------------------------

BETA = 512.

w = np.linspace(-4, 4, 2**12)
plt.close('all')
plot_points = [0.3, 0.5, 0.7, 0.85, 1.1]
U = 2.3
fig_g, axg = plt.subplots(
    len(plot_points), 1, sharex=True, figsize=(6, 1.7 * len(plot_points)))
fig_s, axs = plt.subplots(
    len(plot_points), 1, sharex=True, figsize=(6, 1.7 * len(plot_points)))

for i, tp in enumerate(plot_points):
    sa = ipt_co_s(U, tp, BETA, 'ins', w)
    plot_spectral(w, tp, sa, (axg[i], axs[i]), (-3, 3))
    axg[i].set_ylabel(r'$t_\perp={}$'.format(tp), size=15)
    axs[i].set_ylabel(r'$t_\perp={}$'.format(tp), size=15)

axg[0].set_title(r'$A_B(w)$ @ $U={}$'.format(U))
axs[0].set_title(r'$\Sigma_B(w)$ @ $U={}$'.format(U))
axg[-1].set_xlabel(r'$\omega$')
axs[-1].set_xlabel(r'$\omega$')
axs[-1].set_xlim(-2.5, 2.5)
axg[-1].set_xlim(-2.5, 2.5)

fig_g.subplots_adjust(hspace=0.06)
fig_s.subplots_adjust(hspace=0.06)
# fig_g.savefig('IPT_EF_qp_DOS_tpevol_ins.pdf')
# fig_s.savefig('IPT_EF_qp_sigma_tpevol_ins.pdf')
# plt.close('all')

###############################################################################
# Change from Peierls to Mott insulator by increasing correlation
# ---------------------------------------------------------------

BETA = 512.

w = np.linspace(-4, 4, 2**12)
plt.close('all')
plot_points = [
    (1.4, (-1, 1)),
    (2.4,  (-3, 3)),
    (3.3,  (-8, 8)),
    (4.3,  (-12, 12))]
tp = 0.8

fig_g, axg = plt.subplots(
    len(plot_points), 1, sharex=True, figsize=(6, 1.7 * len(plot_points)))
fig_s, axs = plt.subplots(
    len(plot_points), 1, sharex=True, figsize=(6, 1.7 * len(plot_points)))

for i, (U, ylim) in enumerate(plot_points):
    sa = ipt_co_s(U, tp, BETA, 'ins', w)
    plot_spectral(w, tp, sa, (axg[i], axs[i]), ylim)
    axg[i].set_ylabel(r'$U={}$'.format(U), size=15)
    axs[i].set_ylabel(r'$U={}$'.format(U), size=15)

axg[0].set_title(r'$A_B(w)$ @ $t_\perp={}$'.format(tp))
axs[0].set_title(r'$\Sigma_B(w)$ @ $t_\perp={}$'.format(tp))
axg[-1].set_xlabel(r'$\omega$')
axs[-1].set_xlabel(r'$\omega$')
axs[-1].set_xlim(-4, 4)
axg[-1].set_xlim(-4, 4)
fig_g.subplots_adjust(hspace=0.06)
fig_s.subplots_adjust(hspace=0.06)
# fig_g.savefig('IPT_EF_qp_DOS_Uevol_ins_tp.pdf')
# fig_s.savefig('IPT_EF_qp_sigma_Uevol_ins_tp.pdf')
# plt.close('all')
