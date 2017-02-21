# -*- coding: utf-8 -*-
"""
Thermodynamics of the insulator Dimer lattice
=============================================

Close to the coexistence region the Energy of the system is calculated.
"""
from math import log, ceil
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import dmft.dimer as dimer
import dmft.common as gf


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

BETARANGE = np.around(np.hstack((1 / np.linspace(1e-3, 0.2, 140),
                                 np.logspace(2.3, -4.5, 42, base=2))),
                      decimals=3)

temp = 1 / BETARANGE

U_int = [3.3, 3.7, 4.7, 5.2]
TP = 0.3
avgH = [loop_beta(u_int, TP, BETARANGE, 'I') for u_int in U_int]

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
CV = [np.ediff1d(H) / np.ediff1d(temp) for H in avgH]
for u, cv in zip(U_int, CV):
    plt.plot(temp[:temp_cut], cv[:temp_cut], label='U={}'.format(u))

plt.xlim(-0.1, 2.)
plt.ylim(-0.1, 8.5)
plt.title('Internal Energy')
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
    cv_temp = np.hstack((np.clip(cv, 0, 1) / temp[: -1], 0))
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
