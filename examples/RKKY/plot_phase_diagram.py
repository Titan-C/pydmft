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
