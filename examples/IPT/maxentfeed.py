r"""
test maxIteration
=================

"""
from __future__ import division, absolute_import, print_function

from dmft.ipt_imag import dmft_loop
from dmft.common import greenF, tau_wn_setup, pade_contination, plot_band_dispersion


import numpy as np
import matplotlib.pylab as plt

U = 2.5
beta = 12.5


def ipt_feed(U, beta):
    tau, w_n = tau_wn_setup(dict(BETA=beta, N_MATSUBARA=1024))
    g_iwn0 = greenF(w_n)
    g_iwn, s_iwn = dmft_loop(U, 0.5, -1.j / w_n, w_n, tau, conv=1e-12)
    x = int(2 * beta)
    fw = np.concatenate((-w_n[:x][::-1], w_n[:x]))
    ggiw = np.concatenate((g_iwn.conj()[:x][::-1], g_iwn[:x]))
    sgiw = np.concatenate((s_iwn.conj()[:x][::-1], s_iwn[:x]))
    plt.plot(w_n, g_iwn.imag)

    omega = np.linspace(-3, 3, 600)
    gw = pade_contination(g_iwn, w_n[:x], omega)
    sw = pade_contination(s_iwn, w_n[:x], omega)
    plt.plot(omega, -gw.imag)
    plt.plot(omega, -sw.imag)
    plt.plot(omega, sw.real)
    plt.xlim([0, 3])

    np.savez('/home/oscar/dev/Maxent/ipts_giw_b{}U{}'.format(beta, U),
             w_n=fw, giw=ggiw, std=.003 / (fw**2 + .3**2) + 3e-4, gw_pade=gw, w=omega)
    np.savez('/home/oscar/dev/Maxent/ipts_siw_b{}U{}'.format(beta, U),
             w_n=fw, giw=sgiw, std=.003 / (fw**2 + .3**2) + 3e-4, gw_pade=sw, w=omega)

ipt_feed(2.5, 25)


def Kramers(w, gw):
    """
    #         1
    # K = ----------
    #       w - w'
    """
    dw = w[1:] - w[:-1]
    mw = (w[1:] - w[:-1]) / 2
    dw = np.concatenate((mw, [0])) + np.concatenate(([0], mw))

    K = -dw * gw.imag / np.subtract.outer(w + 1e-14j, w)

    return K.sum(1).real / np.pi


def hiltrans(zeta):
    sqr = np.sqrt(zeta**2 - 1)
    sqr = np.sign(sqr.imag) * sqr
    return 2 * (zeta - sqr)

omega = np.linspace(-5, 5, 600)
gt = hiltrans(omega + 1e-5j)
plt.plot(omega, gt.real, omega, gt.imag)

gr = Kramers(omega, gt)
plt.plot(omega, gr.real)
