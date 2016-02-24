# -*- coding: utf-8 -*-
"""
===================================================================
Phase diagrams of the dimer Bethe Lattice at finite temperature IPT
===================================================================

"""

import numpy as np
import matplotlib.pyplot as plt
import dmft.RKKY_dimer as rt

tpr = np.hstack((np.arange(0, 0.5, 0.02), np.arange(0.5, 1.1, 0.05)))
ur = np.arange(0, 4.5, 0.1)
tpx, uy = np.meshgrid(tpr, ur)

BETARANGE = np.round(np.hstack(([768, 512], np.logspace(8, -4.5, 41, base=2))), 2)
temp = 1 / np.array(BETARANGE)
Tux, Ty = np.meshgrid(ur, temp)

f, ax = plt.subplots(1, 2, figsize=(18, 7))
tpi=12
fl_d = np.load('disk/Dimer_ipt_metal_seed_FL_DOS_BUt.npy')
fli_d = np.load('disk/Dimer_ipt_insulator_seed_FL_DOS_BUt.npy')

s0 = ax[1].pcolormesh(tpx, uy, fl_d[1], cmap=plt.get_cmap('viridis'))
m = ax[0].pcolormesh(Tux, Ty, fl_d[:, :, tpi], cmap=plt.get_cmap('viridis'))


s0 = ax[1].pcolormesh(tpx, uy, fli_d[1], alpha=0.2, cmap=plt.get_cmap('viridis'))
m = ax[0].pcolormesh(Tux, Ty, fli_d[:, :, tpi], alpha=0.2, cmap=plt.get_cmap('viridis'))

ax[1].axis([tpx.min(), tpx.max(), uy.min(), uy.max()])
ax[0].axis([Tux.min(), Tux.max(), 0, .14])

plt.colorbar(m, ax=ax[0])
plt.colorbar(s0, ax=ax[1])

ax[0].set_ylabel('$T/D$')
ax[0].set_xlabel('U/D')
ax[0].set_title('Phase Diagram $t_\\perp={}$\n color represents $-\\Im G_{{AA}}(0)$'.format(tpr[tpi]))
ax[1].set_xlabel(r'$t_\perp$')
ax[1].set_ylabel('U/D')
ax[1].set_title('Phase Diagram $\\beta={}$\n color represents $-\\Im G_{{AA}}(0)$'.format(BETARANGE[1]))

ax[0].plot(2*np.ones_like(temp), temp, 'rx-', lw=2)
ax[0].plot(2.65*np.ones_like(temp), temp, 'rx-', lw=2)
ax[0].plot(3.5*np.ones_like(temp), temp, 'rx-', lw=2)
ax[0].plot(4.3*np.ones_like(temp), temp, 'rx-', lw=2)

ax[1].plot([.2]*4, [2, 2.65, 3.5, 4.3], 'ro', ms=10)


import dmft.common as gf
import dmft.ipt_imag as ipt
def loop_beta(u_int, tp, betarange, seed='mott gap'):
    giw_s = []
    sigma_iw = []
    ekin, epot = [], []
    iterations = []
    for beta in betarange:
        tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5*beta, 256)))
        giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
        if seed == 'mott gap':
            giw_d, giw_o = 1/(1j*w_n - 4j/w_n), np.zeros_like(w_n)+0j

        giw_d, giw_o, loops = rt.ipt_dmft_loop(beta, u_int, -tp, giw_d, giw_o)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(1j*w_n, 1j*giw_d.imag, giw_o.real, 0., -tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, -tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

        ekin.append(rt.ekin(giw_d, giw_o, w_n, beta))

        epot.append(rt.epot(giw_d, giw_o, tau, w_n, -tp, u_int, beta))

    print(np.array(iterations))

    return giw_s, sigma_iw, np.array(ekin), np.array(epot), w_n

giw_s, sigma_iw, ekin, epot, w_n = loop_beta(4, .50, BETARANGE, seed='omott gap')

plt.figure(1)
plt.plot(1/BETARANGE, ekin+epot, '+-')
#plt.plot(1/BETARANGE, epot, '+-')

H=ekin+epot
CV=(H[1:]-H[:-1])/(temp[1:]-temp[:-1])

plt.figure(2)
plt.plot(temp[:-1],CV, '+-')
