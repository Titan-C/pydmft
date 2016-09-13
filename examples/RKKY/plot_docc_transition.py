# -*- coding: utf-8 -*-
"""
=====================================================================
Study the behavior of the Dimer Bethe lattice in the Transition
=====================================================================

Specific Regions of the phase diagram are reviewed to inspect the
behavior of the insulating state """


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.ipt_imag as ipt


def loop_u_tp(u_range, tprange, beta, seed='mott gap'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=256))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'ins':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_s = []
    sigma_iw = []
    ekin, epot = [], []
    iterations = []
    for u_int, tp in zip(u_range, tprange):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

        ekin.append(rt.ekin(giw_d, giw_o, w_n, tp, beta))

        epot.append(rt.epot(giw_d, giw_o, siw_d, siw_o, w_n, tp, u_int, beta))

    print(np.array(iterations))

    return np.array(giw_s), np.array(sigma_iw), np.array(ekin), np.array(epot), w_n

# calculating multiple regions
fac = np.arctan(.55 * np.sqrt(3) / .15)
udelta = np.tan(np.linspace(-fac, fac, 121)) * .15 / np.sqrt(3)
dudelta = np.diff(udelta)
data = []
bet_uc = [(18, 3.312),
          (19, 3.258),
          (20, 3.214),
          (20.5, 3.193),
          (21, 3.17),
          (21.5, 3.1467),
          (21.7, 3.138)]

for beta, uc in bet_uc:
    urange = udelta + uc + .07
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        urange, .3 * np.ones_like(urange), beta, 'met')
    data.append(2 * epot / urange - 0.003)

plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22, 'figure.autolayout': True})
plt.figure()
bc = [b for b, _ in bet_uc]
d_c = [dc[int(len(udelta) / 2)] for dc in data]
for dd, dc, (beta, uc) in zip(data, d_c, bet_uc):
    plt.plot(uc + udelta, dd, '+-', label=r'$\beta={}$'.format(beta))

plt.plot([uc for _, uc in bet_uc], d_c, 'o')

plt.gca().set_color_cycle(None)
for dd, (beta, uc) in zip(data, bet_uc):
    chi = np.diff(dd) / dudelta
    plt.plot(uc + udelta[:-1], chi / np.min(chi) * .035, ':')

plt.title(r'Double occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$U/D$')
plt.legend()
plt.savefig("dimer_tp0.3_docc.pdf",
            transparent=False, bbox_inches='tight', pad_inches=0.05)

# effective scaling
# cubic + linear over constant + linear
plt.figure()


def fit_cube_lin(eta, c, p, q):
    return (c * eta**3 + p * eta) / (1 + q * eta)

for dd, dc, (beta, uc) in zip(data, d_c, bet_uc):
    plt.plot(udelta, dd - dc, lw=2)

plt.gca().set_color_cycle(None)
bb = [30, 30, 35, 42, 45, 48, 50]
for dd, dc, bound, (beta, uc) in zip(data, d_c, bb, bet_uc):
    rd = dd - dc
    popt, pcov = curve_fit(
        fit_cube_lin, rd[bound:-bound], udelta[bound:-bound], p0=[4e4, 3, 3])
    ft = fit_cube_lin(rd, *popt)
    plt.plot(ft, rd)
    plt.plot(ft[bound:-bound], rd[bound:-bound], "k+")
    print(popt)

plt.xlim([-.08, .08])
plt.ylim([-.007, .01])

plt.title(r'Reduced Double occupation fitted to theory')
plt.ylabel(r'$\eta$')
plt.xlabel(r'$U-U_c$')
plt.legend()
plt.savefig("dimer_tp0.3_eta_landau.pdf",
            transparent=False, bbox_inches='tight', pad_inches=0.05)

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
