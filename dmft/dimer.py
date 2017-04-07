# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
Based on the work G. Moeller et all PRB 59, 10, 6846 (1999)
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import h5py
from slaveparticles.quantum import fermion

import dmft.common as gf
import dmft.ipt_imag as ipt
from dmft.utils import optical_conductivity as opt_sig

# Molecule


def sorted_basis(basis_numbering=False):
    """Sorts the basis of states for the electrons in the molecule
    Enforces ordering in particle number an Sz projection"""

    ind = np.array([0, 1, 2, 4, 8, 5, 10, 6, 9, 12, 3, 7, 11, 13, 14, 15])
    if basis_numbering:
        ind = basis_numbering

    basis = [fermion.destruct(4, sigma)[ind][:, ind] for sigma in range(4)]
    return basis


def hamiltonian(u_int, mu, tp, basis_fermions=None):
    r"""Generate an isolated bi-atomic Hamiltonian in particle-hole
    symmetry at mu=0. Include chemical potential for grand Canonical calculations

    .. math::
        \mathcal{H} - \mu N =
        -\frac{U}{2}(n_{a\uparrow} - n_{a\downarrow})^2
        -\frac{U}{2}(n_{b\uparrow} - n_{b\downarrow})^2  +
        t_\perp (a^\dagger_\uparrow b_\uparrow +
                 b^\dagger_\uparrow a_\uparrow +
                 a^\dagger_\downarrow b_\downarrow +
                 b^\dagger_\downarrow a_\downarrow)
        - \mu(n_{a\uparrow} + n_{a\downarrow})
        - \mu(n_{b\uparrow} + n_{b\downarrow})

    Parameters:
        u_int (float): local Coulomb interaction
        mu (float): chemical potential
        tp (float): hopping amplitude between atoms
        basis_fermions (list): 4 element list with sparse matrices
            representing fermion destruction operators
            default basis is [a_up, b_up, a_dw, b_dw]

    Returns
    -------
    h_loc : scipy.sparse.csr.csr_matrix
        Hamiltonian
    basis_fermions : list scipy.sparse.csr.csr_matrix
        fermion desctruction operators in sparse matrix form


    """
    if basis_fermions is None:
        basis_fermions = sorted_basis()

    a_up, b_up, a_dw, b_dw = basis_fermions
    spin_za = a_up.T * a_up - a_dw.T * a_dw
    spin_zb = b_up.T * b_up - b_dw.T * b_dw
    h_loc = - u_int / 2 * spin_za * spin_za - \
        mu * (a_up.T * a_up + a_dw.T * a_dw)
    h_loc += - u_int / 2 * spin_zb * spin_zb - \
        mu * (b_up.T * b_up + b_dw.T * b_dw)
    h_loc += tp * (a_up.T * b_up + a_dw.T * b_dw +
                   b_up.T * a_up + b_dw.T * a_dw)
    return h_loc, [a_up, b_up, a_dw,  b_dw]


def diag_loc_fermions(basis_fermions):
    """Rotate diagonal Fermion matrix operators from diagonal to local basis"""

    from math import sqrt

    as_up, s_up, as_dw, s_dw = basis_fermions
    a_up = (-as_up + s_up) / sqrt(2)
    b_up = (as_up + s_up) / sqrt(2)
    a_dw = (-as_dw + s_dw) / sqrt(2)
    b_dw = (as_dw + s_dw) / sqrt(2)

    return [a_up, b_up, a_dw,  b_dw]


def hamiltonian_diag(u_int, mu, tp, basis_fermions=None):
    r"""Generate an isolated bi-atomic Hamiltonian in particle-hole symmetry at
    mu=0. Include chemical potential for grand Canonical calculations

    This in the diagonal basis [as_up, s_up, as_dw, s_dw]

    See also
    --------
    hamiltonian

    """
    if basis_fermions is None:
        basis_fermions = sorted_basis()

    return hamiltonian(u_int, mu, tp,
                       diag_loc_fermions(basis_fermions))[0], basis_fermions


###############################################################################
# Dimer Bethe lattice


def gf_met(omega, mu, tp, t, tn):
    """Double semi-circular density of states to represent the non-interacting
    dimer """

    g_1 = gf.greenF(omega, mu=mu - tp, D=2 * (t + tn))
    g_2 = gf.greenF(omega, mu=mu + tp, D=2 * abs(t - tn))
    g_d = .5 * (g_1 + g_2)
    g_o = .5 * (g_1 - g_2)

    return g_d, g_o


def mat_inv(a, b):
    """Inverts the relevant entries of the dimer Green's function matrix

    .. math:: [a, b]^-1 = [a, -b]/(a^2 - b^2)
    """
    det = a * a - b * b
    return a / det, -b / det


def mat_2_inv(A):
    """Inverts a 2x2 matrix"""
    det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    return np.asarray([[A[1, 1], -A[0, 1]],  [-A[1, 0],  A[0, 0]]]) / det


def mat_mul(a, b, c, d):
    """Multiplies two Matrices of the dimer Green's Functions"""
    return a * c + b * d, a * d + b * c


def self_consistency(omega, Gd, Gc, mu, tp, t2):
    """Sets the dimer Bethe lattice self consistent condition for the diagonal
    and out of diagonal block
    """

    Dd = omega + mu - t2 * Gd
    Dc = -tp - t2 * Gc

    return mat_inv(Dd, Dc)


def dimer_dyson(g0iw_d, g0iw_o, siw_d, siw_o):
    """Returns Dressed Green Function from G0 and Sigma"""

    sgd, sgo = mat_mul(g0iw_d, g0iw_o, -siw_d, -siw_o)
    sgd += 1.
    dend, dendo = mat_inv(sgd, sgo)

    return mat_mul(dend, dendo, g0iw_d, g0iw_o)


def get_sigmaiw(giw_d, giw_o, w_n, tp):
    """Return Sigma by dyson in paramagnetic case only 2 entries"""

    g0iw_d = 1j * w_n - .25 * giw_d
    g0iw_o = -tp - 0.25 * giw_o

    g_1_iw_d, g_1_iw_o = mat_inv(giw_d, giw_o)
    siw_d = g0iw_d - g_1_iw_d
    siw_o = g0iw_o - g_1_iw_o
    return siw_d, siw_o


def ipt_dmft_loop(BETA, u_int, tp, giw_d, giw_o, tau, w_n, conv=1e-12, t=.5):
    """Self-consistently solve dimer DMFT problem with ipt solver

    Parameters
    ----------
    BETA : float - inverse temperature
    u_int : float - onsite interaction
    tp : float - dimer hybridization
    giw_d : 1D ndarray complex - diagonal Green function G_11
    giw_o : 1D ndarray complex - off-diagonal Green function G_12
    tau : 1D ndarray real - imaginary time array
    w_n : 1D ndarray real - matsubara frequency points
    conv : float - convergence criteria
    t : float - renormalized lattice hopping

    Returns
    -------
    giw_d : 1D ndarray complex - diagonal Green function G_11
    giw_o : 1D ndarray complex - off-diagonal Green function G_12
    loops : int - iterations for converge
"""

    converged = False
    loops = 0
    iw_n = 1j * w_n
    t_sqr = t * t

    while not converged:
        # Half-filling, particle-hole cleaning
        giw_d.real = 0.
        giw_o.imag = 0.

        giw_d_old = giw_d.copy()
        giw_o_old = giw_o.copy()

        g0iw_d, g0iw_o = self_consistency(iw_n, giw_d, giw_o, 0., tp, t_sqr)

        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        giw_d, giw_o = dimer_dyson(g0iw_d, g0iw_o, siw_d, siw_o)

        converged = np.allclose(giw_d_old, giw_d, conv)
        converged *= np.allclose(giw_o_old, giw_o, conv)

        loops += 1
        if loops > 3000:
            converged = True
            print('B', BETA, 'tp', tp, 'U', u_int, 'D', 2 * t)
            print('Failed to converge in less than 3000 iterations')

    return giw_d, giw_o, loops


def ekin(giw_d, giw_o, w_n, tp, beta, t_sqr=0.25):
    r"""Calculates the total kinetic energy of the dimer Bethe lattice

    .. math:: \langle T \rangle = \frac{8}{\beta} \sum_{n>0}
        \left( t_\perp(G_{12}(i\omega_n) -\frac{t_\perp}{(i\omega_n)^2})
            + t^2 ( G_{11}^2 - \frac{1}{(i\omega_n)^2}  + G_{12}^2 ) \right)
            - (t_\perp^2+t^2)\beta

    Parameters
    ----------
    giw_d : complex 1D ndarray
        Diagonal entry of Green's function :math:`G_{11}`
    giw_o : complex 1D ndarray
        Off diagonal entry of Green's function :math:`G_{12}`
    w_n : real 1D ndarray
        Positive Matsubara frequencies
    tp : float
        Dimer hybridization strength
    beta : float
        Inverse temperature
    t_sqr : float
        :math:`t^2` squared hopping to lattice, represent they lattice
        hybridization as :math:`\Delta=t^2G`

    References
    ----------
    :ref:`kinetic_energy`

    See Also
    --------
    epot

    """

    return (tp * giw_o.real + t_sqr * (-giw_d.imag**2 + giw_o.real**2) +
            (t_sqr + tp**2) / w_n**2).sum() / beta * 8 - beta * (t_sqr + tp**2)


def epot(giw_d, w_n, beta, M_3, e_kin, muN):
    r"""Calculates the total potential energy of the dimer

    .. math:: \langle V \rangle = \frac{4}{\beta} \sum_{n>0}
        i\omega_n(G_{11}(i\omega_n) -\frac{1}{i\omega_n} - \frac{M_3}{(i\omega_n)^3})
        - \frac{M_3\beta}{2}+ \frac{\mu}{2}\langle N \rangle - \frac{\langle T \rangle}{2}

    Parameters
    ----------
    giw_d : complex 1D ndarray
        Diagonal entry of Green's function :math:`G_{11}`
    w_n : real 1D ndarray
        Positive Matsubara frequencies
    beta : float
        Inverse temperature
    M_3 : float
        Third Moment in tail expansion of :math:`G_{11}`
    e_kin : float
        Kinetic energy of the dimer :math:`\langle T \rangle`
    muN : float
        Chemical potential weighted by occupation :math:`\mu\langle N \rangle`


    References
    ----------
    :ref:`potential_energy`

    See Also
    --------
    ekin

    """
    return (-w_n * (giw_d.imag + 1 / w_n - M_3 / w_n**3)).sum() * 4 / beta - M_3 * beta / 2 + muN / 2 - e_kin / 2

###############################################################################
# The Symmetric Anti-Symmetric Basis
#


def pade_diag(gf_aa, gf_ab, w_n, w_set, w):
    r"""Take diagonal and off diagonal Matsubara functions in the local
    basis and return real axis functions in the symmetric and
    anti-symmetric basis. Such that

         ⎡ⅈ⋅ωₙ + μ  - t⟂          0       ⎤     ⎡Σ_AA + Σ_AB       0     ⎤
G^{-1} = ⎢                                ⎥  -  ⎢                        ⎥
         ⎣        0        ⅈ⋅ωₙ + μ  + t⟂ ⎦     ⎣     0       Σ_AA - Σ_AB⎦

The Symmetric sum (Anti-bonding) returned first, Asymmetric is returned second

"""

    gf_s = 1j * gf_aa.imag + gf_ab.real  # Anti-bond
    pc = gf.pade_coefficients(gf_s[w_set], w_n[w_set])
    gr_s = gf.pade_rec(pc, w, w_n[w_set])

    gf_a = 1j * gf_aa.imag - gf_ab.real  # bond
    pc = gf.pade_coefficients(gf_a[w_set], w_n[w_set])
    gr_a = gf.pade_rec(pc, w, w_n[w_set])

    return gr_s, gr_a


def _make_gf(h5data):
    """From a dimer Green's Function data entry on hdf5 return a numpy array"""
    pos_giw = np.squeeze(h5data.view(np.complex128))
    return np.concatenate((pos_giw[::-1].conjugate(), pos_giw))


def symdata(gfs_array):
    """Assuming the gfs_array has shape (4, n_freqs)
conjugate the last 2 columns to make the complete array be SYM"""
    gfs_array[2:] = -gfs_array[2:].conjugate()
    return gfs_array


def extract_flat_gf_iter(filename, u_int, last):
    """Returns a list with Green functions of last iterations

    At each iteration there are 4 Green functions sym_up, sym_dw,
    asym_up, asym_dw. Each one of them is extracted individually and
    is an element of the list.

    N.B. Because at half-filling there is particle-hole symmetry the
    sym=-conj(asym).

    Parameters
    ----------
    filename : str
        path to hdf5 file
    u_int : float
        local interaction value
    last : int
        counts the last iterations to extract

    Returns
    -------
    ndarray of shape:  last x 4 x nfreq
    """
    u_str = 'U' + str(u_int)
    block_names = ['sym_up', 'sym_dw', 'asym_up', 'asym_dw']

    with h5py.File(filename, 'r') as datarecord:
        dat = []
        for iteration in list(datarecord[u_str])[-last:]:
            blockgf = datarecord[u_str][iteration]['G_iw']
            dat.append([_make_gf(blockgf[name]['data'].value)
                        for name in block_names])
    return np.array(dat)


def optical_conductivity(beta, ss, sa, omega, tp, eps_k):
    """Calculate the contributions to the optical_conductivity in the dimer

    Parameters
    ----------
    beta : float - Inverse Temperature
    ss : 1D complex ndarray - SYM real-freq self-energy
    sa : 1D complex ndarray - ASYM real-freq self-energy
    omega : 1D float ndarray - real frequency grid (equispaced)
    tp : float - Dimer Hybridization
    eps_k : 1D float ndarray - Energy level bandwidth (equispaced)

    Returns
    -------
    intra_band : Intra band Optical response
    inter_band : Inter band Optical response
    """

    nfp = gf.fermi_dist(omega, beta)
    pos_freq = omega > 0
    rho = np.exp(-2 * eps_k**2) / np.sqrt(np.pi / 2)
    de = eps_k[1] - eps_k[0]
    rhode = (rho * de).reshape(-1, 1)

    def lat_gf(eps_k, w_sig):
        return 1 / np.add.outer(-eps_k, w_sig)

    lat_As = -lat_gf(eps_k, omega - tp - ss + 4e-2j).imag / np.pi
    lat_Aa = -lat_gf(eps_k, omega + tp - sa + 4e-2j).imag / np.pi

    a = opt_sig(lat_Aa, lat_Aa, nfp, omega, rhode)
    a += opt_sig(lat_As, lat_As, nfp, omega, rhode)
    b = opt_sig(lat_Aa, lat_As, nfp, omega, rhode)
    b += opt_sig(lat_As, lat_Aa, nfp, omega, rhode)

    intra_band = a[pos_freq]
    inter_band = b[pos_freq]

    return intra_band, inter_band
