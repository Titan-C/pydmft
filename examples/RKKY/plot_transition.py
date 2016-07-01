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

urange = np.linspace(2.8, 3.4, 300)
data = []
for beta in [16., 18., 20., 22., 24., 26., 28., 30.,  35., 40., 50.]:
    giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
        urange, .3 * np.ones_like(urange), beta, 'met')
    data.append((giw_s, sigma_iw, ekin, epot, w_n, beta))

for sim in data:
    giw_s, sigma_iw, ekin, epot, w_n, beta = sim
    plt.plot(urange, 2 * epot / urange, '-', label=beta)

plt.title(r'Double occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$U/D$')
plt.legend()
plt.show()
