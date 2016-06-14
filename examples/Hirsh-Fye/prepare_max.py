# -*- coding: utf-8 -*-
r"""
=========================
Continuations of HF dimer
=========================

"""
# Created Mon May  2 09:52:26 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import dmft.plot.hf_dimer as pd
import matplotlib.pyplot as plt
import numpy as np
import dmft.common as gf
import os
from scipy.interpolate import splrep, splev, sproot
os.chdir('/home/oscar/dev/dmft-learn/examples/Hirsh-Fye/')


file_root = 'DIMER_PM_B64.0_tp0.3/U2.15/it013/'
seed, beta, u_int = 'met', 64., 2.15
#file_root = 'DIMER_PM_B64.0_tp0.3/U2.16/it010/'
#seed, beta, u_int = 'ins', 64., 2.16

giw_u, gtau_u, tau, w_n, setup = pd.get_giw(file_root + 'gtau_up.npy')
giw_d, gtau_d, tau, w_n, setup = pd.get_giw(file_root + 'gtau_dw.npy')

giw = .5 * (giw_u + giw_d)
gtau = .5 * (gtau_u + gtau_d)

tp = .3

# Redo from the QMC tau, tail following G(tau) discontinuity
gt = gtau.sum(0) * .5
gt = np.concatenate((gt, [-1 - gt[0]]))
taub = np.concatenate((tau, [64.]))

gtautck = splrep(taub, gt, s=0)
dev0 = [splev(0, gtautck, der=i) for i in range(4)]
devb = [splev(64, gtautck, der=i) for i in range(4)]

ders = np.abs(-gf.np.array(dev0) - gf.np.array(devb))

taud = np.arange(0, 64, 64 / 1024.)
plt.plot(taub, gt, taud, splev(taud, gtautck, der=0), '+-')

wnl = gf.matsubara_freq(64., 1024)
gif = gf.gt_fouriertrans(splev(taud, gtautck, der=0), taud, wnl, ders)
# plt.plot(wnl, gif.real * wnl**2, wnl, (gif.imag + 1 / wnl) * wnl**3)
# plt.ylim([-3, 3])

# tail replacement
tail = -1j / wnl - ders[1] / wnl**2 + \
    ders[2] * 1j / wnl**3 + ders[3] / wnl**4
x = 64
gif[x:] = tail[x:]
# plt.plot(wnl, gif.real * wnl**2, wnl, (gif.imag + 1 / wnl) * wnl**3)
plt.figure('G(iwn)')
plt.plot(wnl, gif.real, 'o:', label='metal Re')
plt.plot(wnl, gif.imag, 's:', label='metal Im')


###############################################################################
# Metal
# =====
from glob import glob
sources = sorted(glob(file_root + '*mcs*npy'))
giwss = np.array([np.concatenate((gf.gt_fouriertrans(
    np.load(s).reshape(-1, 128).sum(0) * .25,
    tau, w_n, ders), tail[x:2 * x])) for s in sources]).T
giwf = np.concatenate((giwss[::-1].conj(), giwss))
#ngiwf = np.delete(giwf, [132, 144], 1)
ngiwf = giwf
nwn = np.concatenate((-wnl[:2 * x][::-1], wnl[:2 * x]))
#plt.plot(nwn, ngiwf.imag)
#plt.plot(nwn, ngiwf.real)
sst = ngiwf.std(1)
plt.plot(nwn, sst)
avgiw = ngiwf.mean(1)
sst[sst < 2e-3] = 2e-3
nsiw = (1j * nwn - .3 - .25 * avgiw).reshape(-1, 1) - 1 / ngiwf
avsiw = nsiw.mean(1)
ssiw = nsiw.std(1)
ssiw[ssiw < .01] = .005
plt.plot(nwn, sst)
plt.plot(nwn, .0007 / (nwn**2 + .2**2) + 2e-4)
omega = np.linspace(-4, 4, 600)
gw = gf.pade_continuation(avgiw, nwn, omega, np.arange(128, 200))
sw = gf.pade_continuation(avsiw, nwn, omega, np.arange(128, 200))
plt.plot(omega, gw.imag, lw=3)
plt.plot(omega, sw.imag, lw=3)
np.savez('/home/oscar/dev/Maxent/Dimer_{}_g_b{}U{}'.format(seed, beta, u_int),
         w_n=nwn, giw=avgiw, std=sst, gw_pade=gw, w=omega)
np.savez('/home/oscar/dev/Maxent/Dimer_{}_s_b{}U{}'.format(seed, beta, u_int),
         w_n=nwn, giw=avsiw, std=ssiw, gw_pade=sw, w=omega)
plt.plot(ngiwf[128])
plt.show()
