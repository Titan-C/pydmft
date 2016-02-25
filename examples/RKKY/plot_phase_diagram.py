# -*- coding: utf-8 -*-
"""
===================================================================
Phase diagrams of the dimer Bethe Lattice at finite temperature IPT
===================================================================

"""
from math import log
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.ipt_imag as ipt

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
ax[0].plot(2.75*np.ones_like(temp), temp, 'rx-', lw=2)
ax[0].plot(3.5*np.ones_like(temp), temp, 'rx-', lw=2)
ax[0].plot(4.3*np.ones_like(temp), temp, 'rx-', lw=2)

ax[1].plot([.2]*4, [2, 2.65, 3.5, 4.3], 'ro', ms=10)


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


BETARANGE = np.round(np.hstack(([768, 512], np.logspace(8, -5, 50, base=2))), 2)
temp = 1 / np.array(BETARANGE)

metal2 = loop_beta(2., .20, BETARANGE, seed='metal')
metal2_75 = loop_beta(2.75, .20, BETARANGE, seed='metal')
insulator2_75 = loop_beta(2.75, .20, BETARANGE, seed='mott gap')
metal3_5 = loop_beta(3.5, .20, BETARANGE, seed='metal')
insulator3_5 = loop_beta(3.5, .20, BETARANGE, seed='mott gap')
insulator4_3 = loop_beta(4.3, .20, BETARANGE, seed='mott gap')


###############################################################################
# Internal Energy
# ---------------
#
# It is very strange as $U$ grows into the insulator that there is an
# increase in internal energy as the system is cooled down. It can
# also be seen how the insulator has lower internal energy at $U=3.5$

fig, ax = plt.subplots(2)
Harr = zip([metal2, metal2_75, insulator2_75, metal3_5, insulator3_5, insulator4_3],
           [2, '2.75 M', '2.75 I', '3.5 M', '3.5 I', 4.3],
           ['b', 'g', 'c', 'r', 'y', 'k'])
for sol, u, c in Harr:
    li = c + 'o--' if 'I' in str(u) else c + '+-'
    ax[0].plot(temp, sol[2]+sol[3], li, label='U={}'.format(u))
    ax[1].plot(temp, sol[2]+sol[3], li)
ax[0].set_xlim([0, 5])
ax[0].set_ylim([-0.05, .5])
ax[1].set_xlim([0, .2])
ax[1].set_ylim([-.036, 0.004])

ax[0].set_title('Internal Energy')
ax[0].set_xlabel('T/D')
ax[0].set_ylabel(r'$\langle  H \rangle$')
ax[0].legend(loc=0)
ax[1].set_xlabel('T/D')
ax[1].set_ylabel(r'$\langle  H \rangle$')

###############################################################################
# Specific Heat and Entropy
# -------------------------
#

fig_cv, ax_cv = plt.subplots(2)
fig_s, ax_s = plt.subplots(2)

Harr = zip([metal2, metal2_75, insulator2_75, metal3_5, insulator3_5, insulator4_3],
           [2, '2.75 M', '2.75 I', '3.5 M', '3.5 I', 4.3],
           ['b', 'g', 'c', 'r', 'y', 'k'])
for sol, u, c in Harr:
    H=sol[2]+sol[3]
    CV=(H[1:]-H[:-1])/(temp[1:]-temp[:-1])

    li = c + 'o--' if 'I' in str(u) else c + '+-'
    ax_cv[0].plot(temp[:-1], CV, li, label='U={}'.format(u))
    ax_cv[1].plot(temp[:-1], CV, li)

    cv_temp = np.hstack((np.clip(CV,0,1)/temp[:-1], 0))
    S = np.array([simps(cv_temp[i:], temp[i:], even='last') for i in range(len(temp))])
    ax_s[0].plot(temp, log(2.) -S, li, label='U={}'.format(u))
    ax_s[1].plot(temp, log(2.) -S, li)

ax_cv[0].set_xlim([0, 3])
ax_cv[0].set_ylim([-0.06, .6])
ax_cv[1].set_xlim([0, .125])
ax_cv[1].set_ylim([-0.06, 0.6])

ax_cv[0].set_title('Heat Capacity')
ax_cv[0].set_xlabel('T/D')
ax_cv[0].set_ylabel(r'$C_V$')
ax_cv[1].set_xlabel('T/D')
ax_cv[1].set_ylabel(r'$C_V$')
ax_cv[0].legend(loc=0)


ax_s[0].set_xlim([0, 2])
ax_s[0].set_ylim([0.0, log(2)])
ax_s[1].set_xlim([0, .125])
ax_s[1].set_ylim([0., 0.43])

ax_s[0].set_title('Entropy')
ax_s[0].set_xlabel('T/D')
ax_s[0].set_ylabel(r'$S$')
ax_s[1].set_xlabel('T/D')
ax_s[1].set_ylabel(r'$S$')
ax_s[0].legend(loc=0)
ax_s[0].axhline(log(2)/2, ls=":")
ax_s[1].axhline(log(2)/2, ls=":")
