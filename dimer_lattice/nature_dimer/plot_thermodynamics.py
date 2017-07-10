# -*- coding: utf-8 -*-
"""
Energy calculation of the Dimer lattice
=======================================

Close to the coexistence region the Energy of the system is calculated.
"""
from math import log, ceil
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import dmft.dimer as dimer
import dmft.common as gf
from dmft.utils import differential_weight


tpr = np.hstack((np.arange(0, 0.5, 0.02), np.arange(0.5, 1.1, 0.05)))
ur = np.arange(0, 4.5, 0.1)

tpi = 15  # This sets tp in the calculation tpi=15-> tp=0.3

BETARANGE = np.round(
    np.hstack(([768, 512], np.logspace(8, -4.5, 41, base=2))), 2)
temp = 1 / np.array(BETARANGE)
Tux, Ty = np.meshgrid(ur, temp)
tpx, uy = np.meshgrid(tpr, ur)

f, ax = plt.subplots(1, 2, figsize=(18, 7))
fl_d = np.load('disk/dimer_07_2015/Dimer_ipt_metal_seed_FL_DOS_BUt.npy')
fli_d = np.load(
    'disk/dimer_07_2015/Dimer_ipt_insulator_seed_FL_DOS_BUt.npy')

s0 = ax[1].pcolormesh(tpx, uy, fl_d[1], cmap=plt.get_cmap('viridis'))
m = ax[0].pcolormesh(Tux, Ty, fl_d[:, :, tpi], cmap=plt.get_cmap('viridis'))


s0 = ax[1].pcolormesh(tpx, uy, fli_d[1], alpha=0.2,
                      cmap=plt.get_cmap('viridis'))
m = ax[0].pcolormesh(Tux, Ty, fli_d[:, :, tpi], alpha=0.2,
                     cmap=plt.get_cmap('viridis'))

ax[1].axis([tpx.min(), tpx.max(), uy.min(), uy.max()])
ax[0].axis([Tux.min(), Tux.max(), 0, .14])

plt.colorbar(m, ax=ax[0])
plt.colorbar(s0, ax=ax[1])

ax[0].set_ylabel('$T/D$')
ax[0].set_xlabel('U/D')
ax[0].set_title('Phase Diagram $t_\\perp={}$\n'
                'color represents $-\\Im G_{{AA}}(0)$'.format(tpr[tpi]))
ax[1].set_xlabel(r'$t_\perp$')
ax[1].set_ylabel('U/D')
ax[1].set_title('Phase Diagram $\\beta={}$\n'
                'color represents $-\\Im G_{{AA}}(0)$'.format(BETARANGE[1]))

U_int = [1.8, 2.5, 3.5, 4.3]
for u_int in U_int:
    ax[0].axvline(u_int, lw=2, color='r')

ax[1].plot([tpr[tpi]] * 4, U_int, 'ro', ms=10)

###############################################################################


def loop_beta(u_int, tp, betarange, seed):
    avgH = []
    for beta in betarange:
        tau, w_n = gf.tau_wn_setup(
            dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(8 * beta) / log(2)), 256)))
        giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)
        if seed == 'I':
            giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j

        giw_d, giw_o, _ = dimer.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-6)

        ekin = dimer.ekin(giw_d[:int(8 * beta)], giw_o[:int(8 * beta)],
                          w_n[:int(8 * beta)], tp, beta)

        epot = dimer.epot(giw_d[:int(8 * beta)], w_n[:int(8 * beta)],
                          beta, u_int ** 2 / 4 + tp**2 + 0.25, ekin, u_int)
        avgH.append(ekin + epot)

    return np.array(avgH)

fac = np.arctan(25 * np.sqrt(3) / 0.4)
temp = np.tan(np.linspace(5e-3, fac, 195)) * 0.4 / np.sqrt(3)
BETARANGE = 1 / temp


start = ['M', 'M', 'I', 'I']
avgH = [loop_beta(u_int, tpr[tpi], BETARANGE, seed)
        for u_int, seed in zip(U_int, start)]

###############################################################################
# Internal Energy
# ---------------
#
# The internal energy decreases and upon cooling but in the insulator it
# finds two energy plateaus

plt.figure()
temp_cut = sum(temp < 3)
for u, sol in zip(U_int, avgH):
    plt.plot(temp[:temp_cut], sol[:temp_cut], label='U={}'.format(u))

plt.xlim(0, 2.5)
plt.title('Internal Energy')
plt.xlabel('$T/D$')
plt.ylabel(r'$\langle  H \rangle$')
plt.legend(loc=0)

###############################################################################
# Specific Heat
# -------------
#
# In the heat capacity it is very noticeable how close one is to the
# Quantum Critical Point as the Heat capacity is almost diverging for the
# smallest :math:`U` insulator

plt.figure()
CV = [differential_weight(H) / differential_weight(temp) for H in avgH]
for u, cv in zip(U_int, CV):
    plt.plot(temp[:temp_cut], cv[:temp_cut], label='U={}'.format(u))

plt.xlim(-0.1, 2.)
plt.ylim(-0.1, 8.5)
plt.title('Heat Capacity')
plt.xlabel('$T/D$')
plt.ylabel(r'$C_V$')

###############################################################################
# Entropy
# -------
#
# Entropy again find two plateaus as first evidenced by the internal
# energy, it is also noteworthy that entropy seems to remain finite for
# this insulator and would seem that is the same value for all cases being
# numerical uncertainties to blame for the curves not matching at zero
# temperature. It still remains unknown what the finite entropy value
# corresponds to. In this particula case is :math:`\sim \ln(1.6)`

ENDS = []
for cv in CV:
    cv_temp = np.clip(cv, 0, 1) / temp
    s_t = np.array([simps(cv_temp[i:], temp[i:], even='last')
                    for i in range(len(temp))])
    ENDS.append(log(16.) - s_t)

plt.figure()
for u, s in zip(U_int, ENDS):
    plt.plot(temp[:temp_cut], s[:temp_cut], label='U={}'.format(u))

plt.title('Entropy')
plt.xlabel('$T/D$')
plt.ylabel(r'$S$')

plt.xlim(-0.01, 0.9)
plt.yticks([0, log(2), log(2) * 2, log(2) * 4],
           [0, r'$\ln 2$', r'$2\ln 2$', r'$4\ln 2$'])

plt.show()
