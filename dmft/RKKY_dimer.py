# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
Based on the work G. Moeller et all PRB 59, 10, 6846 (1999)
"""
from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, inverse, TailGf, iOmega_n

from dmft.twosite import matsubara_Z
from joblib import Parallel, delayed
from pytriqs.archive import HDFArchive
from scipy.integrate import quad
import dmft.common as gf
import dmft.ipt_imag as ipt
import dmft.hirschfye as hf
import dmft.h5archive as h5
import numpy as np
import slaveparticles.quantum.dos as dos

###############################################################################
# Dimer Bethe lattice

def gf_met(omega, mu, tp, t, tn):
    """Double semi-circular density of states to represent the
    non-interacting dimer """

    g_1 = gf.greenF(omega, mu=mu-tp, D=2*(t+tn))
    g_2 = gf.greenF(omega, mu=mu+tp, D=2*abs(t-tn))
    g_d = .5*(g_1 + g_2)
    g_o = .5*(g_1 - g_2)

    return g_d, g_o


def mat_inv(a, b):
    """Inverts the relevant entries of the dimer Green's function matrix

    .. math:: [a, b]^-1 = [a, -b]/(a^2 - b^2)
    """
    det = a*a - b*b
    return a/det, -b/det


def mat_mul(a, b, c, d):
    """Multiplies two Matrices of the dimer Green's Functions"""
    return a*c + b*d, a*d + b*c


def self_consistency(omega, Gd, Gc, mu, tp, t2):
    """Sets the dimer Bethe lattice self consistent condition for the diagonal
    and out of diagonal block
    """

    Dd = omega + mu - t2 * Gd
    Dc = -tp - t2 * Gc

    return mat_inv(Dd, Dc)


def mix_gf_dimer(gmix, omega, mu, tab):
    """Dimer formation Green function term

    .. math:: G_{mix}(i\omega_n) =ao
    """
    gmix['A', 'A'] = omega + mu
    gmix['A', 'B'] = -tab
    gmix['B', 'A'] = -tab
    gmix['B', 'B'] = omega + mu
    return gmix


def fit_tail(g_iw):
    """Fits a tail with the known first moment decay as 1/w"""
    fixed_co = TailGf(2, 2, 3, -1)
    fixed_co[1] = np.array([[1, 0], [0, 1]])
    mesh = len(g_iw.mesh)
    g_iw.fit_tail(fixed_co, 8, int(0.65*mesh), mesh)


def init_gf_met(g_iw, omega, mu, tab, tn, t):
    """Gives a metalic seed of a non-interacting system

    """

    Gd, Gc = gf_met(omega, mu, tab, tn, t)
    load_gf_from_np(g_iw, Gd, Gc)

    if isinstance(g_iw, GfImFreq):
        fit_tail(g_iw)


def init_gf_ins(g_iw, omega, U):
    r"""Initializes the green function in the insulator limit given by

    .. math:: G_{11} = (i\omega_n \pm \frac{U^2}{4i\omega_n})^{-1}
    """
    G1 = 1./(1j*omega + U**2 / 4j/omega)
    G2 = 1./(1j*omega - U**2 / 4j/omega)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    load_gf_from_np(g_iw, Gd, Gc)


def load_gf_from_np(g_iw, g_iwd, g_iwo):
    """Loads into the first greenfunction the equal diagonal terms and the
    offdiagonals. Input is numpy array"""

    g_iw['A', 'A'].data[:, 0, 0] = g_iwd
    g_iw['A', 'B'].data[:, 0, 0] = g_iwo
    g_iw['B', 'A'] << g_iw['A', 'B']
    g_iw['B', 'B'] << g_iw['A', 'A']


def load_gf(g_iw, g_iwd, g_iwo):
    """Loads into the first greenfunction the equal diagonal terms and the
    offdiagonals. Input in GF_view"""

    g_iw['A', 'A'] << g_iwd
    g_iw['B', 'B'] << g_iwd
    g_iw['A', 'B'] << g_iwo
    g_iw['B', 'A'] << g_iwo


class Dimer_Solver(object):

    def __init__(self, **params):

        self.U = params['U']
        self.beta = params['BETA']
        self.setup = params

        self.g_iw = GfImFreq(indices=['A', 'B'], beta=self.beta,
                             n_points=params['N_MATSUBARA'])
        self.g0_iw = self.g_iw.copy()
        self.sigma_iw = self.g_iw.copy()

        # Imaginary time
        self.g0_tau = GfImTime(indices=['A', 'B'], beta=self.beta)
        self.sigma_tau = self.g0_tau.copy()

    def solve(self):

        self.g0_tau << InverseFourier(self.g0_iw)
        for name in [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]:
            self.sigma_tau[name].data[:] = self.U**2 * \
                                           self.g0_tau[name].data * \
                                           self.g0_tau[name].data * \
                                           self.g0_tau[name].data[::-1]

        self.sigma_iw << Fourier(self.sigma_tau)

        # Dyson equation to get G
        self.g_iw << inverse(inverse(self.g0_iw) - self.sigma_iw)



def gf_symetrizer(G):
    gd = 1j*np.squeeze(0.5*(G['A', 'A'].data + G['B', 'B'].data)).imag
    gn = np.squeeze(0.5*(G['A', 'B'].data + G['B', 'A'].data)).real
    load_gf_from_np(G, gd, gn)


def dimer(S, gmix, filename, step):

    converged = False
    loops = 0
    t_hop = np.matrix([[S.setup['t'], S.setup['tn']],
                       [S.setup['tn'], S.setup['t']]])
    while not converged:
        # Enforce DMFT Paramagnetic, IPT conditions
        # Pure imaginary GF in diagonals
        S.g_iw.data[:, 0, 0] = 1j*S.g_iw.data[:, 0, 0].imag
        S.g_iw['B', 'B'] << S.g_iw['A', 'A']
        # Pure real GF in off-diagonals
#        S.g_iw.data[:, 0, 1] = S.g_iw.data[:, 1, 0].real
        S.g_iw['B', 'A'] << 0.5*(S.g_iw['A', 'B'] + S.g_iw['B', 'A'])
        S.g_iw['A', 'B'] << S.g_iw['B', 'A']

        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - t_hop * S.g_iw * t_hop
        S.g0_iw.invert()
        S.solve()
#        import pdb; pdb.set_trace()

        converged = np.allclose(S.g_iw.data, oldg, atol=1e-5)
        loops += 1
#        mix = mixer(loops)
        if loops > 2000:
            converged = True

    S.setup.update({'U': S.U, 'loops': loops})

    store_sim(S, filename, step)


def store_sim(S, file_str, step_str):
    file_name = file_str.format(**S.setup)
    step = step_str.format(**S.setup)
    with HDFArchive(file_name, 'a') as R:
        R[step+'setup'] = S.setup
        R[step+'G_iwd'] = S.g_iw['A', 'A']
        R[step+'G_iwo'] = S.g_iw['A', 'B']


def recover_lastit(S, file_str):
    try:
        file_name = file_str.format(**S.setup)
        with HDFArchive(file_name, 'r') as R:
            ru = 'U'+str(S.U)
            lastit = R[ru].keys()[-1]
            load_gf(S.g_iw, R[ru][lastit]['G_iwd'], R[ru][lastit]['G_iwo'])
            S.setup['loops'] = R[ru][lastit]['setup']['loops']
            S.setup['max_loops'] += S.setup['loops']
    except (IOError, KeyError):
        pass


def pot_energy(file_str):
    r"""Calculates the internal energy of the system given by Fetter-Walecka
    25-26
    """

    results = HDFArchive(file_str, 'r')
    ftr_key = results.keys()[0]
    setup = results[ftr_key]['setup']
    beta = setup['beta']
    n_freq = len(results[ftr_key]['G_iwd'].mesh)

    Gfree = GfImFreq(indices=['A', 'B'], beta=beta,
                     n_points=n_freq)

    total_e = []
    Giw = Gfree.copy()
    Siw = Gfree.copy()
    for uint in results:
        load_gf(Giw, results[uint]['G_iwd'], results[uint]['G_iwo'])
        load_gf(Siw, results[uint]['S_iwd'], results[uint]['S_iwo'])
        energ = 0.5*Siw*Giw
        u = results[uint]['setup']['U']+1e-15
        total_e.append(energ.total_density()/u + .25)

    del results

    return np.asarray(total_e)

def total_energy(file_str):
    r"""Calculates the internal energy of the system given by Fetter-Walecka
    25-26

    .. math:: \langle H \rangle = 1/\beta\sum_{nk}
         1/2(i\omega_n +  H^0_k + \mu)
         Tr G(k, i\omega_n)\\
         = Tr 1/2(i\omega_n +  H^0_k + \mu)G - G^{0,-1}G^0 + G^{-1}G)\\
         = Tr 1/2(i\omega_n +  H^0_k + \mu)G - (i\omega_n - H^0_k + \mu)G^0 +
          (i\omega_n - H^0_k +\mu -\Sigma )G)
         = Tr i\omega_n(G-G^0)

    """

    results = HDFArchive(file_str, 'r')
    ftr_key = results.keys()[0]
    setup = results[ftr_key]['setup']
    beta, tab, tn, t = setup['beta'], setup['tab'], setup['tn'], setup['t']
    n_freq = len(results[ftr_key]['G_iwd'].mesh)

    Gfree = GfImFreq(indices=['A', 'B'], beta=beta,
                     n_points=n_freq)
    w_n = gf.matsubara_freq(beta, n_freq)
    om_id = mix_gf_dimer(Gfree.copy(), iOmega_n, 0., 0.)
    init_gf_met(Gfree, w_n, 0., tab, tn, t)

    mean_free_ekin = quad(dos.bethe_fermi_ene, -2*t, 2*t,
                          args=(1., tab, t, beta))[0] \
                     - tab*quad(dos.bethe_fermi, -tab, tab,
                                args=(1., 0., t, beta))[0]

    total_e = []
    Giw = Gfree.copy()
    Siw = Gfree.copy()
    for uint in results:
        load_gf(Giw, results[uint]['G_iwd'], results[uint]['G_iwo'])
        load_gf(Siw, results[uint]['S_iwd'], results[uint]['S_iwo'])
        energ = om_id * (Giw - Gfree) - 0.5*Siw*Giw
        total_e.append(energ.total_density() + 2*mean_free_ekin)

    del results

    return np.asarray(total_e)

def complexity(filestr):
    """Extracts the loopcount for convergence"""
    with h5.File(filestr, 'r') as results:
        comp = [results[uint]['loops'].value for uint in results]
    return np.asarray(comp)


def quasiparticle(filestr, beta, tp):
    zet = []
    with h5.File(filestr, 'r') as results:
        tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=5*beta))
        iw_n = 1j*w_n
        for ustr in results:
            g0d, g0o = self_consistency(iw_n,
                                        results[ustr]['giw_d'][:],
                                        results[ustr]['giw_o'][:],
                                        0., tp, 0.25)

            u_int = float(ustr[1:])
            g0td, g0to = ipt.gw_fourier(g0d, g0o, tau, w_n, u_int, tp)
            st_d, _ = ipt.dimer_sigma(g0td, g0to, u_int)
            sw_d = gf.gt_fouriertrans(st_d, tau, w_n, [u_int**2/4, 0., u_int**2/4])


            zet.append(matsubara_Z(sw_d.imag, beta))
            print(u_int, sw_d[:2], zet[-1])
    return np.asarray(zet)


def fermi_level_dos(filestr, beta, n=3):
    with h5.File(filestr, 'r') as results:
        w_n = gf.matsubara_freq(beta, n)
        fl_dos = [gf.fit_gf(w_n, results[uint]['giw_d'][:n].imag)(0.) for uint in results]
    return np.asarray(fl_dos)


def proc_files(filelist):
    """Extracts the diffulty, quasiparticle weigth, fermi_lev dos, and Energy

    Parameters
    ----------
    filelist:
        list that contains the paths to files to be processed

    Returns
    -------
    4-tuple of 2D ndarrays. First axis corresponds to filelist, second to file
    data. Keep in mind H5 files return keys in alphabetical and not write order
    """

    dif = np.asarray(Parallel(n_jobs=-1)(delayed(complexity)(f)
                                         for f in filelist))
    zet = np.asarray(Parallel(n_jobs=-1)(delayed(quasiparticle)(f)
                                         for f in filelist))
    imet = np.asarray(Parallel(n_jobs=-1)(delayed(fermi_level_dos)(f)
                                          for f in filelist))
    H = np.asarray(Parallel(n_jobs=-1)(delayed(total_energy)(f)
                                       for f in filelist))

    return dif, zet, imet, H


def result_pros(tabra, beta):
    filelist = ['met_fuloop_t0.5_tab{}_B{}.h5'.format(it, beta)
                for it in tabra]
    met_sol = proc_files(filelist)
    filelist = ['ins_fuloop_t0.5_tab{}_B{}.h5'.format(it, beta)
                for it in tabra]
    ins_sol = proc_files(filelist)

    np.savez('results_fuloop_t0.5_B{}'.format(beta),
             difm=met_sol[0], zetm=met_sol[1], imetm=met_sol[2], Hm=met_sol[3],
             difi=ins_sol[0], zeti=ins_sol[1], imeti=ins_sol[2], Hi=ins_sol[3])
