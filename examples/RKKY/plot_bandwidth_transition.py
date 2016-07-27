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
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.ipt_imag as ipt


def loop_u_tp(Drange, tprange, beta, seed='mott gap'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=256))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'mott gap':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_s = []
    sigma_iw = []
    ekin, epot = [], []
    iterations = []
    for D, tp in zip(Drange, tprange):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, 1, tp, giw_d, giw_o, tau, w_n, t=D / 2)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, D**2 / 4)
        siw_d, siw_o = ipt.dimer_sigma(1, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

        ekin.append(rt.ekin(giw_d, giw_o, w_n, tp, beta))

        epot.append(rt.epot(giw_d, giw_o, siw_d, siw_o, w_n, tp, 1, beta))

    print(np.array(iterations))

    return np.array(giw_s), np.array(sigma_iw), np.array(ekin), np.array(epot), w_n

Drange = np.linspace(0.05, .85, 61)
data = []
for beta in [16., 18., 20., 22., 24., 26., 28.]:
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        Drange, .3 * np.ones_like(Drange), beta, 'met')
    data.append((giw_s, sigma_iw, ekin, epot, w_n, beta))

for sim in data:
    giw_s, sigma_iw, ekin, epot, w_n, beta = sim
    plt.plot(Drange, 2 * epot, '-', label=beta)

plt.title(r'Double occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$D/U$')
plt.legend()
