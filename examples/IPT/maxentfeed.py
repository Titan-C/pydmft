r"""
test maxIteration
=================

"""
from __future__ import division, absolute_import, print_function

from dmft.ipt_imag import dmft_loop
from dmft.common import greenF, tau_wn_setup, pade_continuation, plot_band_dispersion
import dmft.common as gf
import dmft.RKKY_dimer as rt
import dmft.ipt_imag as ipt


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
    gw = pade_continuation(g_iwn, w_n[:x], omega)
    sw = pade_continuation(s_iwn, w_n[:x], omega)
    plt.plot(omega, -gw.imag)
    plt.plot(omega, -sw.imag)
    plt.plot(omega, sw.real)
    plt.xlim([0, 3])

    np.savez('/home/oscar/dev/pymaxent/ipts_giw_b{}U{}'.format(beta, U),
             w_n=fw, giw=ggiw, std=.003 / (fw**2 + .3**2) + 3e-4, gw_pade=gw, w=omega)
    np.savez('/home/oscar/dev/pymaxent/ipts_siw_b{}U{}'.format(beta, U),
             w_n=fw, giw=sgiw, std=.003 / (fw**2 + .3**2) + 3e-4, gw_pade=sw, w=omega)

ipt_feed(2.5, 64)


def ipt_dimer(u_int, tp, beta, seed):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=1024))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'ins':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_d, giw_o, loops = rt.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-12)
    g0iw_d, g0iw_o = rt.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    x = int(2 * beta)
    fw = np.concatenate((-w_n[:x][::-1], w_n[:x]))
    g_iwn = giw_o.real + 1j * giw_d.imag
    ggiw = np.concatenate((g_iwn.conj()[:x][::-1], g_iwn[:x]))
    s_iwn = siw_o.real + 1j * siw_d.imag
    sgiw = np.concatenate((s_iwn.conj()[:x][::-1], s_iwn[:x]))
    plt.plot(w_n, g_iwn.imag)

    omega = np.linspace(-3, 3, 600)
    gw = pade_continuation(g_iwn, w_n[:x], omega)
    sw = pade_continuation(s_iwn, w_n[:x], omega)
    plt.plot(omega, -gw.imag)
    plt.plot(omega, -sw.imag)
    plt.plot(omega, sw.real)
    plt.xlim([0, 3])

    np.savez('/home/oscar/dev/pymaxent/ipts_dim_{}_giw_b{}U{}'.format(seed, beta, u_int),
             w_n=fw, giw=ggiw, std=.003 / (fw**2 + .3**2) + 3e-4, gw_pade=gw, w=omega)
    np.savez('/home/oscar/dev/pymaxent/ipts_dim_{}_siw_b{}U{}'.format(seed, beta, u_int),
             w_n=fw, giw=sgiw, std=.003 / (fw**2 + .3**2) + 3e-4, gw_pade=sw, w=omega)

ipt_dimer(2., .8, 64, 'met')
ipt_dimer(2.5, .3, 64, 'ins')


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


omega = np.linspace(-5, 5, 600)
gt = gf.semi_circle_hiltrans(omega + 1e-5j)
plt.plot(omega, gt.real, omega, gt.imag)

gr = Kramers(omega, gt)
plt.plot(omega, gr.real)

sigw = rw - .3 - .25 * gw - 1 / gw
