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
import numpy as np
from pytriqs.archive import HDFArchive
import dmft.common as gf
import slaveparticles.quantum.dos as dos
from scipy.integrate import quad
from dmft.twosite import matsubara_Z
import dmft.hirschfye as hf
from joblib import Parallel, delayed


def mix_gf_dimer(gmix, omega, mu, tab):
    """Dimer formation Green function term

    .. math::

        G_{mix}(i\omega_n) =ao
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
    G1 = gf.greenF(omega, mu=mu-tab, D=2*(t+tn))
    G2 = gf.greenF(omega, mu=mu+tab, D=2*abs(t-tn))
    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    load_gf_from_np(g_iw, Gd, Gc)

    if isinstance(g_iw, GfImFreq):
        fit_tail(g_iw)


def init_gf_ins(g_iw, omega, U):
    """Initializes the green function in the insulator limit given by

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
                             n_points=params['n_points'])
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

class Dimer_Solver_hf(Dimer_Solver):

    def __init__(self, **params):
        super(Dimer_Solver_hf, self).__init__(**params)
        self.g0_tau = GfImTime(indices=['A', 'B'], beta=self.beta,
                               n_points=self.setup['N_TAU'])
        self.g_tau = self.g0_tau.copy()

    def patch_tail(self):
        fixed_co = TailGf(2, 2, 4, -1)
        fixed_co[1] = np.array([[1, 0], [0, 1]])
        fixed_co[2] = self.setup['tp']*np.array([[0, 1], [1, 0]])
        self.g_iw.fit_tail(fixed_co, 4, self.setup['n_tau_mc']//2, self.setup['n_points'])

    def solve(self, tau):

        self.g0_tau << InverseFourier(self.g0_iw)
        g0t = np.asarray([self.g0_tau(t).real for t in tau])

        gtu, gtd = hf.imp_solver(g0t, g0t, self.V_field, self.setup)
        gt_D = -0.25 * (gtu[0, 0] + gtu[1, 1] + gtd[0, 0] + gtd[1, 1])
        gt_N = -0.25 * (gtu[1, 0] + gtu[0, 1] + gtd[1, 0] + gtd[0, 1])

        gt_D = hf.interpol(gt_D, self.setup['N_TAU']-1)
        gt_N = hf.interpol(gt_N, self.setup['N_TAU']-1)

        load_gf_from_np(self.g_tau, gt_D, gt_N)

        self.g_iw << Fourier(self.g_tau)
        self.patch_tail()


def gf_symetrizer(G):
    gd = 1j*np.squeeze(0.5*(G['A', 'A'].data + G['B', 'B'].data)).imag
    gn = np.squeeze(0.5*(G['A', 'B'].data + G['B', 'A'].data)).real
    load_gf_from_np(G, gd, gn)


def dimer(S, gmix, filename, step):

    converged = False
    loops = 0
    t_hop = np.matrix([[S.setup['t'], S.setup['tn']],[S.setup['tn'], S.setup['t']]])
    while not converged:
        # Enforce DMFT Paramagnetic, IPT conditions
        # Pure imaginary GF in diagonals
        S.g_iw.data[:, 0, 0] = 1j*S.g_iw.data[:, 0, 0].imag
        S.g_iw['B', 'B']<<S.g_iw['A', 'A']
        # Pure real GF in off-diagonals
#        S.g_iw.data[:, 0, 1] = S.g_iw.data[:, 1, 0].real
        S.g_iw['B', 'A']<< 0.5*(S.g_iw['A', 'B'] + S.g_iw['B', 'A'])
        S.g_iw['A', 'B']<<S.g_iw['B', 'A']

        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - t_hop * S.g_iw * t_hop
        S.g0_iw.invert()
        S.solve()
#        import pdb; pdb.set_trace()

        converged = np.allclose(S.g_iw.data, oldg, atol=1e-3)
        loops += 1
#        mix = mixer(loops)
        if loops > 2000:
            converged = True

    S.setup.update({'U': S.U, 'loops': loops})

    store_sim(S, filename, step)


def store_sim(S, file_str, step_str):
    file_name = file_str.format(**S.setup)
    step = step_str.format(**S.setup)
    R = HDFArchive(file_name, 'a')
    R[step+'setup'] = S.setup
    R[step+'G_iwd'] = S.g_iw['A', 'A']
    R[step+'G_iwo'] = S.g_iw['A', 'B']
#    R[step+'S_iwd'] = S.sigma_iw['A', 'A']
#    R[step+'S_iwo'] = S.sigma_iw['A', 'B']
    del R


def total_energy(file_str):
    """Calculates the internal energy of the system given by Fetter-Walecka
    25-26

    .. math:: \langle H \rangle = 1/\beta\sum_{nk} 1/2(i\omega_n +  H^0_k + \mu)
    Tr G(k, i\omega_n)\\
    = Tr 1/2(i\omega_n +  H^0_k + \mu)G - G^{0,-1}G^0 + G^{-1}G)\\
    = Tr 1/2(i\omega_n +  H^0_k + \mu)G - (i\omega_n - H^0_k + \mu)G^0 + (i\omega_n - H^0_k +\mu -\Sigma )G)
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
                  -tab*quad(dos.bethe_fermi, -tab, tab,
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


def complexity(file_str):
    """Extracts the loopcount for convergence"""
    results = HDFArchive(file_str, 'r')
    dif = []
    for uint in results:
        nl = results[uint]['setup']
        dif.append(nl['loops'])
    del results
    return np.asarray(dif)


def quasiparticle(file_str):
    results = HDFArchive(file_str, 'r')
    zet = []
    for uint in results:
        S_iw = results[uint]['S_iwd']
        zet.append(matsubara_Z(S_iw.data[:, 0, 0].imag, S_iw.beta))
    del results
    return np.asarray(zet)


def fit_dos(w_n, g):
    """Performs a quadratic fit of the -first- matsubara frequencies
    to estimate the value at zero energy"""
    n = len(w_n)
    gfit = g.data[:n, 0, 0].imag
    pf = np.polyfit(w_n, gfit, 2)
    return np.poly1d(pf)


def fermi_level_dos(file_str, n=5):
    results = HDFArchive(file_str, 'r')
    ftr_key = results.keys()[0]
    fl_dos = []
    w_n = gf.matsubara_freq(results[ftr_key]['G_iwd'].beta, n)
    for uint in results:
        fl_dos.append(abs(fit_dos(w_n, results[uint]['G_iwd'])(0.)))
    del results
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

    dif = np.asarray(Parallel(n_jobs=-1)(delayed(complexity)(f) for f in filelist))
    zet = np.asarray(Parallel(n_jobs=-1)(delayed(quasiparticle)(f) for f in filelist))
    imet = np.asarray(Parallel(n_jobs=-1)(delayed(fermi_level_dos)(f) for f in filelist))
    H = np.asarray(Parallel(n_jobs=-1)(delayed(total_energy)(f) for f in filelist))

    return dif, zet, imet, H


def result_pros(tabra, beta):
    filelist = ['met_fuloop_t0.5_tab{}_B{}.h5'.format(it, beta) for it in tabra]
    met_sol = proc_files(filelist)
    filelist = ['ins_fuloop_t0.5_tab{}_B{}.h5'.format(it, beta) for it in tabra]
    ins_sol = proc_files(filelist)

    np.savez('results_fuloop_t0.5_B{}'.format(beta),
             difm=met_sol[0], zetm=met_sol[1], imetm=met_sol[2], Hm=met_sol[3],
             difi=ins_sol[0], zeti=ins_sol[1], imeti=ins_sol[2], Hi=ins_sol[3])
