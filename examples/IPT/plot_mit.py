# -*- coding: utf-8 -*-
r"""
===================================
The Metal Mott Insulator transition
===================================

Using a real frequency IPT solver follow the spectral function along
the metal to insulator transition.

"""
from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.signal as signal
import matplotlib.pylab as plt

from slaveparticles.quantum.operators import fermi_dist
import dmft.common as gf
import dmft.ipt_real as ipt


def dmft_loop(gloc, w, u_int, beta, conv):
    """DMFT Loop for the single band Hubbard Model at Half-Filling


    Parameters
    ----------
    gloc : complex 1D ndarray
        local Green's function to use as seed
    w : real 1D ndarray
        real frequency points
    u_int : float
        On site interaction, Hubbard U
    beta : float
        Inverse temperature
    conv : float
        convergence criteria

    Returns
    -------
    gloc : complex 1D ndarray
        DMFT iterated local Green's function
    sigma : complex 1D ndarray
        DMFT iterated self-energy

"""

    dw = w[1] - w[0]
    eta = 2j * dw
    nf = fermi_dist(w, beta)

    converged = False
    while not converged:

        gloc_old = gloc.copy()
        # Self-consistency
        g0 = 1 / (w + eta - .25 * gloc)
        # Spectral-function of Weiss field
        A0 = -g0.imag / np.pi

        # Second order diagram
        isi = ipt.ph_hf_sigma(A0, nf, u_int) * dw * dw
        isi = 0.5 * (isi + isi[::-1])

        # Kramers-Kronig relation, uses Fourier Transform to speed convolution
        hsi = -signal.hilbert(isi, len(isi) * 4)[:len(isi)].imag
        sigma = hsi + 1j * isi

        # Semi-circle Hilbert Transform
        gloc = gf.semi_circle_hiltrans(w - sigma)
        converged = np.allclose(gloc, gloc_old, atol=conv)

    return gloc, sigma

w = np.linspace(-4, 4, 2**12)
gloc = gf.semi_circle_hiltrans(w + 1e-3j)

urange = [0.2, 1., 2., 3., 3.5, 4.]
plt.close('all')
for i, U in enumerate(urange):
    gloc, sigma_loc = dmft_loop(gloc, w, U, 400, 1e-5)

    plt.gca().set_prop_cycle(None)
    shift = -2.1 * i
    plt.plot(w, shift + -gloc.imag)
    plt.axhline(shift, color='k', lw=0.5)

plt.xlabel(r'$\omega$')
plt.xlim([-4, 4])
plt.ylim([shift, 2.1])
plt.yticks(0.5 - 2.1 * np.arange(len(urange)), ['U=' + str(u) for u in urange])
