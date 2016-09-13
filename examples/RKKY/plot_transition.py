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
    if seed == 'mott gap':
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
fac = np.arctan(.15 * np.sqrt(3) / .10)
udelta = np.tan(np.linspace(-fac, fac, 91)) * .10 / np.sqrt(3)
dudelta = np.diff(udelta)
data = []
bet_uc = [(18, 2.312),
          (19, 2.339),
          (20, 2.3638),
          (20.5, 2.375),
          (21, 2.386)]

for beta, uc in bet_uc:
    urange = udelta + uc + .07
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        urange, .0 * np.ones_like(urange), beta, 'met')
    data.append(2 * epot / urange - 0.003)

plt.figure()
bc = [b for b, _ in bet_uc]
plt.figure()
for dd, dc, (beta, uc) in zip(data, d_c, bet_uc):
    plt.plot(uc + udelta, dd, '+-', label=beta)

plt.plot([uc for _, uc in bet_uc], [dc[45] for dc in data], 'o')

for dd, (beta, uc) in zip(data, bet_uc):
    chi = np.diff(dd) / dudelta
    plt.plot(uc + udelta[:-1], chi / np.min(chi) * .035, ':')

# effective scaling
plt.close()
plt.figure()
d_c = [dc[45] for dc in data]
for dd, dc, (beta, uc) in zip(data, d_c, bet_uc):
    plt.plot(udelta, dd - dc, lw=2)


def fit_cube(eta, c):
    return c * eta**3

plt.gca().set_color_cycle(None)
for dd, dc, (beta, uc) in zip(data, d_c, bet_uc):
    rd = dd - dc
    bound = 35
    popt, pcov = curve_fit(fit_cube, rd[bound:-bound], udelta[bound:-bound])
    ft = fit_cube(rd, *popt)
    plt.plot(ft, rd)
    print(popt)

plt.xlim([-.15, .15])

# cubic + linear


def fit_cube_lin(eta, c, p):
    return c * eta**3 + p * eta

for dd, dc, (beta, uc) in zip(data, d_c, bet_uc):
    plt.plot(udelta, dd - dc, lw=2)

plt.gca().set_color_cycle(None)
bb = [23, 27, 33, 33, 36]
for dd, dc, bound, (beta, uc) in zip(data, d_c, bb, bet_uc):
    rd = dd - dc
    popt, pcov = curve_fit(
        fit_cube_lin, rd[bound:-bound], udelta[bound:-bound])
    ft = fit_cube_lin(rd, *popt)
    plt.plot(ft, rd)
    plt.plot(ft[bound:-bound], rd[bound:-bound], "k+")
    print(popt)

plt.xlim([-.15, .15])
# cubic + linear over constant + linear
plt.figure()


def fit_cube_lin(eta, c, p, q, s):
    return (c * eta**3 + p * eta + s) / (1 + q * eta)

for dd, dc, (beta, uc) in zip(data, d_c, bet_uc):
    plt.plot(udelta, dd - dc, lw=2)

plt.gca().set_color_cycle(None)
bb = [20, 26, 33, 33, 37]
for dd, dc, bound, (beta, uc) in zip(data, d_c, bb, bet_uc):
    rd = dd - dc
    popt, pcov = curve_fit(
        fit_cube_lin, rd[bound:-bound], udelta[bound:-bound], p0=[4e4, 0, 3, -.3])
    ft = fit_cube_lin(rd, *popt)
    plt.plot(ft, rd)
    plt.plot(ft[bound:-bound], rd[bound:-bound], "k+")
    print(popt)

plt.xlim([-.15, .15])
# Beta 21 study


def fitchi(urange, a, uc, x0):
    return a * np.abs(urange - uc)**(-2 / 3) + x0

ulim = 44
popt, pcov = curve_fit(fitchi, urange[:ulim], chi[
                       :ulim], p0=[-0.01, 2.391, -0.01])
ft = fitchi(urange, *popt)
plt.plot(urange, ft)
plt.plot(urange[:ulim], ft[:ulim], lw=2)
print(popt)

ulim += 4
popt, pcov = curve_fit(fitchi, urange[ulim:], chi[ulim:], p0=[-0.01, 2.391, 0])
ft = fitchi(urange, *popt)
plt.plot(urange, ft)
plt.plot(urange[ulim:], ft[ulim:], lw=2)
print(popt)

plt.figure()
beta = 21
giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
    urange + .07, .0 * np.ones_like(urange), beta, 'met')
dd = 2 * epot / urange - 0.003
#fi = np.polyfit(dd[::-1], urange[::-1], 3)
#uf = np.poly1d(fi)(dd[::-1])
ucl = 44


def func(eta, Uc, p, h, k, hp):
    return Uc + h * (eta**3 + k * eta + hp) / (1 + p * eta)
popt, pcov = curve_fit(func, dd[:ucl], urange[:ucl], p0=[2.385, 1, 5, 1, 0])
print(popt)
uf = func(dd[:ucl], *popt)
plt.plot(urange, dd, '+-')
plt.plot(uf, dd[:ucl], 'o:')
ucl = 46
popt, pcov = curve_fit(func, dd[ucl:], urange[ucl:])
print(popt)
uf = func(dd[ucl:], *popt)
plt.plot(uf, dd[ucl:], 'o:')


def func(eta, p, c, h, d):
    return p * eta + c * eta**3 - h + d * eta**2

popt, pcov = curve_fit(func, dd, urange)
print(popt)
uf = func(dd, *popt)
plt.plot(uf, dd, 'o:')
plt.plot(urange, dd)

popt, pcov = curve_fit(func, dd[:ucl], urange[:ucl])
print(popt)
uf = func(dd, *popt)
plt.plot(uf, dd, 'o:')

ucl = 30
popt, pcov = curve_fit(func, dd[ucl:], urange[ucl:])
print(popt)
uf = func(dd, *popt)
plt.plot(uf, dd, 'o:')

plt.title(r'Double occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$U/D$')
plt.legend()
plt.show()


# beta 20 study
beta = 20
giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
    urange + .07, .0 * np.ones_like(urange), beta, 'met')
dd = 2 * epot / urange - 0.003
