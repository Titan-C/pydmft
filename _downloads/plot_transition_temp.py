# -*- coding: utf-8 -*-
"""
=====================================================================
Study the behavior of the Dimer Bethe lattice in the Transition
=====================================================================

Specific Regions of the phase diagram are reviewed to inspect the
behavior of the insulating state """


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import dmft.dimer as dimer
import dmft.common as gf
import dmft.ipt_imag as ipt


def loop_u_tp(u_int, tp, betarange, seed='mott gap'):

    giw_s = []
    sigma_iw = []
    ekin, epot = [], []
    iterations = []
    lwn = []
    for beta in betarange:
        tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=512))
        lwn.append(w_n)

        giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)
        if seed == 'mott gap':
            giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

        giw_d, giw_o, loops = dimer.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = dimer.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

        ekin.append(dimer.ekin(giw_d, giw_o, w_n, tp, beta))

        epot.append(dimer.epot(giw_d, w_n, beta,
                               u_int ** 2 / 4 + tp**2, ekin[-1], u_int))
    print(np.array(iterations))
    # last division in energies because I want per spin epot
    return np.array(giw_s), np.array(sigma_iw), np.array(ekin) / 4, np.array(epot) / 4, w_n

urange = np.arange(2.8, 3.4, .1)
data = []
temp = np.linspace(0.015, 0.06, 90)
betarange = 1 / temp

for u_int in urange:
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        u_int, .3, betarange, 'mott gap')
    data.append((giw_s, sigma_iw, ekin, epot, w_n, u_int))

for u_int in urange:
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        u_int, .3, betarange, 'met')
    data.append((giw_s, sigma_iw, ekin, epot, w_n, u_int))

for sim in data:
    giw_s, sigma_iw, ekin, epot, w_n, u_int = sim
    plt.plot(1 / betarange, 2 * epot / u_int, 'x-', label=u_int)

plt.title(r'Double occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$T/D$')
plt.legend()
plt.show()
