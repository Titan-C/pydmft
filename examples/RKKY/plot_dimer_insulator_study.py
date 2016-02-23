# -*- coding: utf-8 -*-
"""
=====================================================================
Study the behavior of the Dimer Bethe lattice in the Insulator region
=====================================================================

Specific Region of the phase diagram are reviewed to inspect the
behavior of the insulating state """


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.ipt_imag as ipt


def loop_u_tp(u_range, tprange, beta, seed='mott gap'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5*beta, 256)))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'mott gap':
        giw_d, giw_o = 1/(1j*w_n + 4j/w_n), np.zeros_like(w_n)+0j

    giw_s = []
    sigma_iw = []
    ekin, epot = [], []
    iterations = []
    for u_int, tp in zip(u_range, tprange):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(beta, u_int, tp, giw_d, giw_o)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(1j*w_n, 1j*giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

        ekin.append((-giw_d.imag**2 + giw_o.real**2 + 1/w_n**2).real.sum()/beta/2 - beta/16)

        epot.append((-siw_d.imag*giw_d.imag + siw_o.real*giw_o.real + u_int**2/4/w_n**2).real.sum()/beta - beta*u_int**2/32 + u_int/8)


    print(np.array(iterations))

    return np.array(giw_s), np.array(sigma_iw), np.array(ekin), np.array(epot), w_n

###############################################################################

tpr = np.hstack((np.arange(0, 0.5, 0.02), np.arange(0.5, 1.1, 0.05)))
ur = np.arange(0, 4.5, 0.1)
x, y = np.meshgrid(tpr, ur)
BETA = 512.
filestr = 'disk/Dimer_ipt_B{}.h5'.format(BETA)
metal_phases = np.clip(-rt.fermi_level_dos(filestr, BETA).T , 0, 2)
filestr = 'disk/Dimer_ins_ipt_B{}.h5'.format(BETA)
insulator_phases = np.clip(-rt.fermi_level_dos(filestr, BETA).T, 0, 2)


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
giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(2.65*np.ones_like(tprr), tprr, 512.)

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2, cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title('Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))
plt.plot(tprr, 2.65*np.ones_like(tprr), 'rx-', lw=2)

###############################################################################
# Change of G_{AA}
# -------------
#
# There is not much out of the ordinary here. An always gaped function

#BETA = 512.
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=max(5*BETA, 256)))
#X, Y = np.meshgrid(np.arange(0, 1.2, 0.04), w_n[:500])
#ax.plot_surface(X, Y, giw_s[:, 0,:500].imag.T)
#

slopes = [np.polyfit(w_n[:5], giw_s[i, 0,:5].imag, 1)[0] for i in range(len(tprr))]
plt.plot(tprr, slopes, '+-', label='data')
plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")
#plt.plot(tprr, (slopes[-1]-slopes[0])/tprr[-1]*tprr+slopes[0], ':')


###############################################################################
# Analytical Conts
# ----------------

wSet=np.concatenate((np.arange(5)*512, np.arange(1, 241, 1)))
w=np.linspace(0, 3.5, 500)
plt.figure()
for i in range(len(tprr)):
    pc= gf.pade_coefficients(1j*giw_s[i, 0, wSet].imag, w_n[wSet])
    plt.plot(w, tprr[i] - gf.pade_rec(pc, w, w_n[wSet]).imag)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega) + t_\perp$')


###############################################################################
# Change of G_{AB}
# ----------------

plt.plot(w_n, giw_s[:, 1].real.T)
plt.xlim([0, 3])


plt.figure()

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1,:5].real, 1) for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)


###############################################################################
# Sigma AA
# --------

plt.figure()
plt.plot(w_n, sigma_iw[:, 0].T.imag, '-')
plt.xlim([0,.3])
plt.ylim([-8, 0])

###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 0,:5].imag, 1) for i in range(1, len(tprr))])

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
plt.plot(w_n, sigma_iw[:, 1].T.real, '+-')
plt.xlim([0, 3.2])
plt.ylim([-0.1, 2])

###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 1, :5].real, 1) for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Re \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(tprr, ekin+epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# Following $U_{c1}$
# =============
#
# Here I simultaneously vary $U$ and $t_\perp$ just to stay above $U_{c1}$

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2, cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title('Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))
plt.plot(tprr, 2.8-.65/.4*tprr, 'rx-', lw=2)

giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(2.8-.65/.4*tprr, tprr, 512.)


###############################################################################
# Change of $G_{AA}$
# ------------------


BETA = 512.
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.arange(0, 1.2, 0.04), w_n[:500])
ax.plot_surface(X, Y, giw_s[:, 0, :500].imag.T)


###############################################################################

slopes = [np.polyfit(w_n[:5], giw_s[i, 0, :5].imag, 1)[0] for i in range(len(tprr))]
plt.plot(tprr, slopes, '+-', label='data')

plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")


###############################################################################
# Analytical Conts
# ----------------

wSet=np.concatenate((np.arange(5)*512, np.arange(1, 241, 1)))
w=np.linspace(0, 3.5, 500)
plt.figure()
for i in range(len(tprr)):
    pc= gf.pade_coefficients(1j*giw_s[i, 0, wSet].imag, w_n[wSet])
    plt.plot(w, -i/10 - gf.pade_rec(pc, w, w_n[wSet]).imag)


###############################################################################
# Change of $G_{AB}$
# ------------------

plt.plot(w_n, giw_s[:, 1].real.T, 'x:')
plt.xlim([0, 3])


###############################################################################

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1,:5].real, 1) for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)

###############################################################################
# Sigma AA
# --------

plt.plot(w_n, sigma_iw[:, 0].T.imag, '+-')
plt.xlim([0,.3])
plt.ylim([-8, 0])

###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 0,:5].imag, 1) for i in range(1, len(tprr))])

plt.figure()
plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Im \Sigma_{AA}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Im \Sigma_{AA}(0)$')
plt.legend(loc=0)


###############################################################################
# Simga AB
# --------

plt.figure()
plt.plot(w_n, sigma_iw[:, 1].T.real, '+-')
plt.xlim([0, 3.2])
plt.ylim([-0.1, 2])

###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 1,:5].real, 1) for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Re \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(tprr, ekin+epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# High in the Mott insulator
# ==========================

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2, cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title('Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))
plt.plot(tprr, 4.3*np.ones_like(tprr), 'rx-', lw=2)

giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(4.3*np.ones_like(tprr), tprr, 512.)


###############################################################################
# Change of $G_{AA}$
# ------------------

BETA = 512.
fig = plt.figure()
ax = fig.gca(projection='3d')
tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=max(5*BETA, 256)))
X, Y = np.meshgrid(np.arange(0, 1.2, 0.04), w_n[:500])
ax.plot_surface(X, Y, giw_s[:, 0,:500].imag.T)


slopes = [np.polyfit(w_n[:5], giw_s[i, 0,:5].imag, 1)[0] for i in range(len(tprr))]
plt.plot(tprr, slopes, '+-', label='data')

plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")

###############################################################################
# Analytical Conts
# ----------------

wSet=np.concatenate((np.arange(5)*512, np.arange(1, 241, 1)))
w=np.linspace(0, 3.5, 500)
plt.figure()
for i in range(len(tprr)):
    pc= gf.pade_coefficients(1j*giw_s[i, 0, wSet].imag, w_n[wSet])
    plt.plot(w, -i/10 - gf.pade_rec(pc, w, w_n[wSet]).imag)


###############################################################################
# Change of $G_{AB}$
# ------------------


plt.plot(w_n, giw_s[:, 1].real.T, 'x:')
plt.xlim([0, 3])

###############################################################################

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1,:5].real, 1) for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)

###############################################################################
# Sigma AA
# --------

plt.figure()
plt.plot(w_n, sigma_iw[:, 0].T.imag, '+-')
plt.xlim([0,.3])
plt.ylim([-8, 0])

###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 0,:5].imag, 1) for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Im \Sigma_{AA}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Im \Sigma_{AA}(0)$')
plt.legend(loc=0)

###############################################################################
# Simga AB
# --------

plt.figure()
plt.plot(w_n, sigma_iw[:, 1].T.real, '+-')
plt.xlim([0, 3.2])
plt.ylim([-0.1, 2])


###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 1,:5].real, 1) for i in range(1, len(tprr))])

plt.plot(tprr[1:], slopes.T[0], '+-', label='Slope')
plt.plot(tprr[1:], slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Re \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(tprr, ekin+epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$t_\perp/D$')

###############################################################################
# High in the Band insulator
# ==========================

plt.pcolormesh(x, y, metal_phases, cmap=plt.get_cmap(r'viridis'))
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.pcolormesh(x, y, insulator_phases, alpha=0.2, cmap=plt.get_cmap(r'viridis'))

plt.xlabel(r'$t_\perp$')
plt.ylabel(r'U/D')
plt.title('Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(BETA))

urange = np.linspace(0, 4.5, len(tprr))
plt.plot(1.02*np.ones_like(urange), urange, 'rx-', lw=2)

giw_s, sigma_iw, ekin, epot, w_n = loop_u_tp(urange, 1.02*np.ones_like(urange), 512.)

###############################################################################
# Change of $G_{AA}$
# ------------------
BETA = 512.
fig = plt.figure()
ax = fig.gca(projection='3d')
tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=max(5*BETA, 256)))
X, Y = np.meshgrid(urange, w_n[:500])
ax.plot_surface(X, Y, giw_s[:, 0,:500].imag.T)

###############################################################################

slopes = [np.polyfit(w_n[:5], giw_s[i, 0,:5].imag, 1)[0] for i in range(len(tprr))]
plt.plot(tprr, slopes, '+-', label='data')

plt.title(r'Slope of $Im G_{AA}$')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Im m G'_{AA}(0)$")

###############################################################################
# Analytical Conts
# ----------------

wSet=np.concatenate((np.arange(5)*512, np.arange(1, 220, 1)))
w=np.linspace(0, 3.5, 500)
plt.figure()
for i in range(len(tprr)):
    pc= gf.pade_coefficients(1j*giw_s[i, 0, wSet].imag, w_n[wSet])
    plt.plot(w, -i/10 - gf.pade_rec(pc, w, w_n[wSet]).imag)

###############################################################################
# Change of $G_{AA}$
# ------------------
plt.plot(w_n, giw_s[:, 1].real.T, 'x:')
plt.xlim([0, 3])

###############################################################################

slopes = np.array([np.polyfit(w_n[:5], giw_s[i, 1, :5].real, 1) for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re G_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r"$\Re e G'_{AB}(0)$")
plt.legend(loc=0)

###############################################################################
# Sigma AA
# --------

plt.figure()
a=plt.plot(w_n, sigma_iw[:, 0].T.imag, '+-')
###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 0, :5].imag, 1) for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Im \Sigma_{AA}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Im \Sigma_{AA}(0)$')
plt.legend(loc=0)

###############################################################################
# Sigma AB
# --------

plt.figure()
a=plt.plot(w_n, sigma_iw[:, 1].T.real, '+-')
###############################################################################

slopes = np.array([np.polyfit(w_n[:5], sigma_iw[i, 1, :5].real, 1) for i in range(len(tprr))])

plt.plot(tprr, slopes.T[0], '+-', label='Slope')
plt.plot(tprr, slopes.T[1], '+-', label='Cut')

plt.title(r'Slope of $\Re \Sigma_{AB}$ and zero frequency cut')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$\Re \Sigma_{AB}(0)$')
plt.legend(loc=0)

###############################################################################
# The Energy
# ----------

plt.plot(tprr, ekin+epot)
plt.title(r'Internal Energy per spin')
plt.ylabel(r'$\langle H \rangle$')
plt.xlabel(r'$t_\perp/D$')
