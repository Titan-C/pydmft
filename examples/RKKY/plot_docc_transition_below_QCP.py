# -*- coding: utf-8 -*-
"""
====================================
Landau Theory of the Mott transition
====================================

Perform a fit of the order parameter, linked to double occupation to
match a Landau theory formulation in correspondence to Kotliar, G.,
Lange, E., & Rozenberg, M. J. (2000). Landau Theory of the Finite
Temperature Mott Transition. Phys. Rev. Lett., 84(22),
5180–5183. http://dx.doi.org/10.1103/PhysRevLett.84.5180

Study above the critical point
"""


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import dmft.dimer as dimer
import dmft.common as gf
import dmft.ipt_imag as ipt

plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22, 'figure.autolayout': True})


def dmft_solve(giw_d, giw_o, u_int, tp, beta, tau, w_n):
    giw_d, giw_o, loops = dimer.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n)
    g0iw_d, g0iw_o = dimer.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    ekin = dimer.ekin(giw_d, giw_o, w_n, tp, beta)

    # last division because I want per spin epot
    epot = dimer.epot(giw_d, w_n, beta, u_int ** 2 /
                      4 + tp**2, ekin, u_int) / 4
    return (giw_d, giw_o), (siw_d, siw_o), ekin, epot, loops


def loop_u_tp(u_range, tprange, beta, seed='mott gap'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=256))
    giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'ins':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    sols = [dmft_solve(giw_d, giw_o, u_int, tp, beta, tau, w_n)
            for u_int, tp in zip(u_range, tprange)]
    giw_s = np.array([g[0] for g in sols])
    sigma_iw = np.array([g[1] for g in sols])
    ekin = np.array([g[2] for g in sols])
    epot = np.array([g[3] for g in sols])
    iterations = np.array([g[4] for g in sols])

    print(iterations)

    return giw_s, sigma_iw, ekin, epot, w_n

# below the critical point
fac = np.arctan(.55 * np.sqrt(3) / .25)
udelta = np.tan(np.linspace(-fac, fac, 71)) * .15 / np.sqrt(3)
udelta = udelta[:int(len(udelta) / 2) + 10]
dudelta = np.diff(udelta)
databm = []
databi = []

bet_ucm = [(24, 3.064),
           (28, 3.028),
           (30, 3.023),
           ]

for beta, uc in bet_ucm:
    urange = udelta + uc + .07
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        urange, .3 * np.ones_like(urange), beta, 'met')
    databm.append(2 * epot / urange - 0.003)

bet_uci = [(24, 3.01),
           (28, 2.8),
           (30, 2.719),
           ]

databi = []
for beta, uc in bet_uci:
    urange = -udelta + uc + .07
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        urange, .3 * np.ones_like(urange), beta, 'ins')
    databi.append(2 * epot / urange - 0.003)

plt.figure()
for dd, (beta, uc) in zip(databm, bet_ucm):
    plt.plot(uc + udelta, dd, '+-', label=r'$\beta={}$'.format(beta))

plt.gca().set_color_cycle(None)
d_c = [dc[int(len(udelta) / 2)] for dc in databi]
for dd, dc, (beta, uc) in zip(databi, d_c, bet_uci):
    plt.plot(uc - udelta, dd, '+-')  # , label=r'$\beta={}$'.format(beta))

plt.title(r'Double occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$U/D$')
plt.legend()
plt.savefig("dimer_tp0.3_docc_bTc.pdf",
            transparent=False, bbox_inches='tight', pad_inches=0.05)


def fit_cube_lin(eta, c, p, q, d):
    return (c * (eta - d)**3 + p * (eta - d) + q)  # / (1 + q * eta)

u_cen = [0.027, 0.114, 0.152]
plt.figure()
plt.gca().set_color_cycle(None)
bc = [b for b, _ in bet_ucm]
for ddm, ddi, uc, beta in zip(databm, databi, u_cen, bc):
    cent = (ddm[-11] + ddi[-11]) / 2
    plt.plot(udelta + uc, ddm - cent, '-', lw=2,
             label=r'$\beta={}$'.format(beta))

plt.gca().set_color_cycle(None)
for ddm, ddi, uc in zip(databm, databi, u_cen):
    cent = (ddm[-11] + ddi[-11]) / 2
    plt.plot(-uc - udelta, ddi - cent, '-', lw=2)

plt.gca().set_color_cycle(None)
bb = [9, 9, 9]
lb = 12
bc = [b for b, _ in bet_uci]
for dd, ddi, bound, beta, uc in zip(databm, databi, bb, bc, u_cen):
    cent = (dd[-11] + ddi[-11]) / 2
    popt, pcov = curve_fit(
        fit_cube_lin, dd[lb:-bound], udelta[lb:-bound], p0=[4e4, -17, -.1, 0.035])
    ft = fit_cube_lin(dd, *popt) + uc
    plt.plot(ft[:-bound], dd[:-bound] - cent)
    #plt.plot(ft[lb:-bound], dd[lb:-bound], "k+")
    plt.plot(ft[lb:-bound], dd[lb:-bound] - cent, "k+-")
    print(popt)


def fit_cube_lin(eta, c, p, q, d):
    return (c * (eta - d)**3 + p * (eta - d))  # / (1 + q * eta)

plt.gca().set_color_cycle(None)
bb = [9, 10, 9]
lb = 10
for ddm, dd, bound, uc in zip(databm, databi, bb, u_cen):
    cent = (ddm[-11] + dd[-11]) / 2
    popt, pcov = curve_fit(
        fit_cube_lin, dd[lb:-bound], -udelta[lb:-bound], p0=[-4e4, 7, 0, 0.035])
    ft = fit_cube_lin(dd, *popt) - uc
    plt.plot(ft[:-bound], dd[:-bound] - cent)
    plt.plot(ft[lb:-bound], dd[lb:-bound] - cent, "k+-")
    print(popt)

plt.title(r'Reduced Double occupation fitted to theory')
plt.ylabel(r'$\eta$')
plt.xlabel(r'$U-U_c$')
plt.legend()

plt.savefig("dimer_tp0.3_eta_landau_bTc.pdf",
            transparent=False, bbox_inches='tight', pad_inches=0.05)
