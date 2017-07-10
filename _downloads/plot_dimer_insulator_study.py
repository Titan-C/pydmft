# -*- coding: utf-8 -*-
"""
=====================================================================
Study the behavior of the Dimer Bethe lattice in the Insulator region
=====================================================================

Specific Regions of the phase diagram are reviewed to inspect the
behavior of the insulating state """


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import dmft.dimer as dimer
import dmft.common as gf
import dmft.ipt_imag as ipt


def loop_u_tp(u_range, tprange, beta, seed='mott gap'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
    giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'mott gap':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_s = []
    sigma_iw = []
    ekin, epot = [], []
    iterations = []
    for u_int, tp in zip(u_range, tprange):
        giw_d, giw_o, loops = dimer.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = dimer.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

        ekin.append(dimer.ekin(giw_d, giw_o, w_n, tp, beta))

        epot.append(dimer.epot(giw_d, w_n, beta, u_int **
                               2 / 4 + tp**2, ekin[-1], u_int) / 4)  # last division because I want per spin epot

    print(np.array(iterations))

    return np.array(giw_s), np.array(sigma_iw), np.array(ekin), np.array(epot), w_n

###############################################################################

tpr = np.hstack((np.arange(0, 0.5, 0.02), np.arange(0.5, 1.1, 0.05)))
ur = np.arange(0, 4.5, 0.1)
x, y = np.meshgrid(tpr, ur)
BETA = 512.
metal_phases = np.load(
    'disk/dimer_07_2015/Dimer_ipt_metal_seed_FL_DOS_BUt.npy')[1]
insulator_phases = np.load(
    'disk/dimer_07_2015/Dimer_ipt_insulator_seed_FL_DOS_BUt.npy')[1]


###############################################################################
# Starting just above $U_{c1}$
# ============================
#
# Here I increase the strength of the dimer bond starting from the
# Mott insulating solution for a single impurity problem. I need to
# emphasize that there appears not to exist a connection between the
# finite dimer bonding strength and the decoupled impurities case. A
# metallic states seems to appear in between if they are close
# enough. To cure this behavior one needs to go to ever colder
# temperatures each time.

tprr = np.arange(0, 1.2, 0.04)

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2,
               cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title(
    'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))
plt.plot(tprr, 2.65 * np.ones_like(tprr), 'rx-', lw=2)

###############################################################################

giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
    2.65 * np.ones_like(tprr), tprr, 512.)

###############################################################################
# Change of G_{AA}
# ----------------
#
# There is not much out of the ordinary here. An always gaped function

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.arange(0, 1.2, 0.04), w_n[:500])
ax.plot_surface(X, Y, giw_s[:, 0, :500].imag.T)
ax.set_xlabel(r'$t_\perp/D$')
ax.set_ylabel(r'$i\omega_n$')
ax.set_zlabel(r'$\Im m G_{AA} (i\omega_n)$')

###############################################################################

slopes = [np.polyfit(w_n[:5], giw_s[i, 0, :5].imag, 1)[0]
          for i in range(len(tprr))]
plt.plot(tprr, slopes, '+-', label='data')
plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")

###############################################################################
# Analytical Continuation
# -----------------------

w_set = np.concatenate((np.arange(8) * 256, np.arange(1, 241, 1)))
w = np.linspace(0, 3., 500)
plt.figure()
for i, tp in enumerate(tprr):
    pc = gf.pade_coefficients(1j * giw_s[i, 0, w_set].imag, w_n[w_set])
    plt.plot(w, -2 * tp - gf.pade_rec(pc, w, w_n[w_set]).imag)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega) - 2 t_\perp$')


###############################################################################
# Change of G_{AB}
# ----------------

for i in [1, 2, 4, 7, 11, 16]:
    plt.plot(w_n, giw_s[i, 1].real, 's-',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e G_{AB} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, 3])


plt.figure()

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1, :5].real, 1)
                   for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)


###############################################################################
# Sigma AA
# --------

for i in range(6):
    plt.plot(w_n, sigma_iw[i, 0].imag, 'o:',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, .5])
plt.ylim([-8, 0])

plt.figure()
for i in range(6):
    plt.loglog(w_n, -sigma_iw[i, 0].imag, 'o:',
               label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$-\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 0, :3].imag, 1)
                   for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Im \Sigma_{AA}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Im \Sigma_{AA}(0)$')
plt.legend(loc=0)


###############################################################################
# Sigma AB
# --------

plt.figure()
for i in [1, 4, 9, 16, 21, 25]:
    plt.plot(w_n, sigma_iw[i, 1].real, 's-',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.xlim([0, 3.2])
plt.ylim([-0.1, 2])
plt.legend(loc=0)

plt.figure()
for i in [1, 4, 9, 16, 21, 25]:
    plt.loglog(w_n, sigma_iw[i, 1].real, 's-',
               label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 1, :3].real, 1)
                   for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Re e \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(tprr, ekin + epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# Double occupation
# -----------------

plt.plot(tprr, 2 * epot / 2.65)
plt.title(r'Double Occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# Following $U_{c1}$
# ==================
#
# Here I simultaneously vary $U$ and $t_\perp$ just to stay above $U_{c1}$

giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(2.8 - .65 / .4 * tprr, tprr, 512.)

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2,
               cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title(
    'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))
plt.plot(tprr, 2.8 - .65 / .4 * tprr, 'rx-', lw=2)

###############################################################################
# Change of G_{AA}
# ----------------
#
# There is not much out of the ordinary here. An always gaped function

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.arange(0, 1.2, 0.04), w_n[:500])
ax.plot_surface(X, Y, giw_s[:, 0, :500].imag.T)
ax.set_xlabel(r'$t_\perp/D$')
ax.set_ylabel(r'$i\omega_n$')
ax.set_zlabel(r'$\Im m G_{AA} (i\omega_n)$')

###############################################################################

slopes = [np.polyfit(w_n[:5], giw_s[i, 0, :5].imag, 1)[0]
          for i in range(len(tprr))]
plt.plot(tprr, slopes, '+-', label='data')
plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")


###############################################################################
# Analytical Continuation
# -----------------------

w_set = np.concatenate((np.arange(7) * 256, np.arange(1, 237, 1)))
w = np.linspace(0, 2.5, 500)
plt.figure()
for i, tp in enumerate(tprr):
    pc = gf.pade_coefficients(1j * giw_s[i, 0, w_set].imag, w_n[w_set])
    plt.plot(w, -2 * tp - gf.pade_rec(pc, w, w_n[w_set]).imag)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega) - 2 t_\perp$')


###############################################################################
# Change of G_{AB}
# ----------------

for i in [1, 2, 4, 7, 11, 16]:
    plt.plot(w_n, giw_s[i, 1].real, 's-',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e G_{AB} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, 3])


plt.figure()

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1, :5].real, 1)
                   for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)


###############################################################################
# Sigma AA
# --------

for i in range(6):
    plt.plot(w_n, sigma_iw[i, 0].imag, 'o:',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, 2.5])
plt.ylim([-9, 0])

plt.figure()
for i in range(6):
    plt.loglog(w_n, -sigma_iw[i, 0].imag, 'o:',
               label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'-$\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 0, :3].imag, 1)
                   for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Im \Sigma_{AA}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Im \Sigma_{AA}(0)$')
plt.legend(loc=0)


###############################################################################
# Sigma AB
# --------

plt.figure()
for i in [1, 4, 9, 16, 21, 25]:
    plt.plot(w_n, sigma_iw[i, 1].real, 's-',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.xlim([0, 3.2])
plt.ylim([-0.1, 2])
plt.legend(loc=0)

plt.figure()
for i in [1, 4, 9, 16, 21, 25]:
    plt.loglog(w_n, sigma_iw[i, 1].real, 's-',
               label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 1, :3].real, 1)
                   for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Re e \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(tprr, ekin + epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# Double occupation
# -----------------
plt.plot(tprr, 2 * epot / (2.8 - .65 / .4 * tprr))
plt.title(r'Double Occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# High in the Mott insulator
# ==========================

giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
    4.3 * np.ones_like(tprr), tprr, 512.)

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2,
               cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title(
    'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))
plt.plot(tprr, 4.3 * np.ones_like(tprr), 'rx-', lw=2)

###############################################################################
# Change of G_{AA}
# ----------------
#
# There is not much out of the ordinary here. An always gaped function

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.arange(0, 1.2, 0.04), w_n[:500])
ax.plot_surface(X, Y, giw_s[:, 0, :500].imag.T)
ax.set_xlabel(r'$t_\perp/D$')
ax.set_ylabel(r'$i\omega_n$')
ax.set_zlabel(r'$\Im m G_{AA} (i\omega_n)$')

###############################################################################

slopes = [np.polyfit(w_n[:5], giw_s[i, 0, :5].imag, 1)[0]
          for i in range(len(tprr))]
plt.plot(tprr, slopes, '+-', label='data')
plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")


###############################################################################
# Analytical Continuation
# -----------------------

w_set = np.concatenate((np.arange(7) * 256, np.arange(1, 237, 1)))
w = np.linspace(0.5, 3.5, 500)
plt.figure()
for i, tp in enumerate(tprr):
    pc = gf.pade_coefficients(1j * giw_s[i, 0, w_set].imag, w_n[w_set])
    plt.plot(w, -2 * tp - gf.pade_rec(pc, w, w_n[w_set]).imag)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega) - 2 t_\perp$')


###############################################################################
# Change of G_{AB}
# ----------------

for i in [1, 2, 4, 7, 11, 16]:
    plt.plot(w_n, giw_s[i, 1].real, 's-',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e G_{AB} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, 3])


plt.figure()

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1, :5].real, 1)
                   for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)


###############################################################################
# Sigma AA
# --------

for i in range(6):
    plt.plot(w_n, sigma_iw[i, 0].imag, 'o:',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, 2.5])
plt.ylim([-25, 0])

plt.figure()
for i in range(6):
    plt.loglog(w_n, -sigma_iw[i, 0].imag, 'o:',
               label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$-\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 0, :3].imag, 1)
                   for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Im \Sigma_{AA}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Im \Sigma_{AA}(0)$')
plt.legend(loc=0)


###############################################################################
# Sigma AB
# --------

plt.figure()
for i in [1, 4, 9, 16, 21, 25]:
    plt.plot(w_n, sigma_iw[i, 1].real, 's-',
             label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.xlim([0, 3.2])
plt.ylim([-0.1, 6])
plt.legend(loc=0)


plt.figure()
for i in [1, 4, 9, 16, 21, 25]:
    plt.loglog(w_n, sigma_iw[i, 1].real, 's-',
               label=r'$t_\perp={}$'.format(tprr[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 1, :3].real, 1)
                   for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Re e \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(tprr, ekin + epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# Double occupation
# -----------------
plt.plot(tprr, 2 * epot / 4.3)
plt.title(r'Double Occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# High in the Band insulator
# ==========================

urange = np.linspace(0.5, 4.5, len(tprr))
giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(
    urange, 1.02 * np.ones_like(urange), 512.)

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2,
               cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title(
    'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))

plt.plot(1.02 * np.ones_like(urange), urange, 'rx-', lw=2)

###############################################################################
# Change of G_{AA}
# ----------------
#
# There is not much out of the ordinary here. An always gaped function

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(urange, w_n[:500])
ax.plot_surface(X, Y, giw_s[:, 0, :500].imag.T)
ax.set_xlabel(r'$U/D$')
ax.set_ylabel(r'$i\omega_n$')
ax.set_zlabel(r'$\Im m G_{AA} (i\omega_n)$')

###############################################################################

slopes = [np.polyfit(w_n[:5], giw_s[i, 0, :5].imag, 1)[0]
          for i in range(len(urange))]
plt.plot(urange, slopes, '+-', label='data')
plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$U/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")


###############################################################################
# Analytical Continuation
# -----------------------

w_set = np.concatenate((np.arange(7) * 256, np.arange(1, 247, 1)))
w = np.linspace(0, 3.5, 500)
plt.figure()
for i, u in enumerate(urange):
    pc = gf.pade_coefficients(1j * giw_s[i, 0, w_set].imag, w_n[w_set])
    plt.plot(w, -u - gf.pade_rec(pc, w, w_n[w_set]).imag)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega) - U$')


###############################################################################
# Change of G_{AB}
# ----------------

for i in [3, 7, 11, 16, 23, 28]:
    plt.plot(w_n, giw_s[i, 1].real, 's-', label=r'$U={:.2}$'.format(urange[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e G_{AB} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, 3])


plt.figure()

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1, :5].real, 1)
                   for i in range(len(urange))])

plt.plot(urange, slopes.T[0], '+-', label='Slope')
plt.plot(urange, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$U/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)


###############################################################################
# Sigma AA
# --------

for i in [3, 7, 11, 16, 23, 28]:
    plt.plot(w_n, sigma_iw[i, 0].imag, 'o:',
             label=r'$U={:.2}$'.format(urange[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)
plt.xlim([0, 3.2])

plt.figure()
for i in [3, 7, 11, 16, 23, 28]:
    plt.loglog(w_n, -sigma_iw[i, 0].imag, 'o:',
               label=r'$U={:.2}$'.format(urange[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$-\Im m \Sigma_{AA} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 0, :3].imag, 1)
                   for i in range(len(urange))])

plt.plot(urange, slopes.T[0], '+-', label='Slope')
plt.plot(urange, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Im \Sigma_{AA}$ and zero frequency cut')
plt.xlabel(r'$U/D$')
plt.ylabel(r'$\Im \Sigma_{AA}(0)$')
plt.legend(loc=0)


###############################################################################
# Sigma AB
# --------

plt.figure()
for i in [3, 7, 11, 16, 23, 28]:
    plt.plot(w_n, sigma_iw[i, 1].real, 's-',
             label=r'$U={:.2}$'.format(urange[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.xlim([0, 3.2])
plt.ylim([-0.1, 2])
plt.legend(loc=0)

plt.figure()
for i in [3, 7, 11, 16, 23, 28]:
    plt.loglog(w_n, sigma_iw[i, 1].real, 's-',
               label=r'$U={:.2}$'.format(urange[i]))
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$\Re e \Sigma_{AB} (i\omega_n)$')
plt.legend(loc=0)

###############################################################################

slopes = np.array([np.polyfit(w_n[:3], sigma_iw[i, 1, :3].real, 1)
                   for i in range(len(urange))])

plt.plot(urange, slopes.T[0], '+-', label='Slope')
plt.plot(urange, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$U/D$')
plt.ylabel(r'$\Re e \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(urange, ekin + epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$U/D$')

###############################################################################
# Double occupation
# -----------------
plt.plot(urange, 2 * epot / urange)
plt.title(r'Double occupation')
plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')
plt.xlabel(r'$U/D$')
