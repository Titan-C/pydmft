# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse, TailGf
from pytriqs.gf.local import GfReFreq
from pytriqs.plot.mpl_interface import oplot
from pytriqs.archive import HDFArchive
from dmft.plot.hf_single_site import label_convergence
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, tp=0.25, filestr='DIMER_PM_B{BETA}_tp{tp}.h5',
              block=2, n_freq=2, xlim=2, skip=0, sig=False):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    freq_arr = []
    with HDFArchive(filestr.format(tp=tp, BETA=beta), 'r') as datarecord:
        for step in datarecord[u_str].keys()[skip:]:
            labels = [name for name in datarecord[u_str][step]['G_iw'].indices]
            gf_iw = datarecord[u_str][step]['G_iw']
            u_int = float(u_str[1:])
            paramagnetic_hf_clean(gf_iw, float(u_str[1:]), tp)
            gf_iw = gf_iw[labels[block]]
            if sig:
                shift = 1. if 'asym' in labels[block] else -1
                gf_iw << iOmega_n + u_int / 2. + shift * \
                    tp - 0.25 * gf_iw - inverse(gf_iw)

            axes[0].oplot(gf_iw.imag, 'bo:', label=None)
            axes[0].oplot(gf_iw.real, 'gs:', label=None)

            gf_iw = np.squeeze([gf_iw(i) for i in range(n_freq)])
            freq_arr.append([gf_iw.real, gf_iw.imag])
    freq_arr = np.asarray(freq_arr).T
    for num, (rfreqs, ifreqs) in enumerate(freq_arr):
        axes[1].plot(rfreqs, 's-.', label=str(num))
        axes[1].plot(ifreqs, 'o-.', label=str(num))

    graf = r'$G$' if not sig else r'$\Sigma$'
    graf += r'$(i\omega_n)$ ' + labels[block]
    label_convergence(beta, u_str + '\n$t_\\perp={}$'.format(tp),
                      axes, graf, n_freq, xlim)


def list_show_conv(BETA, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5',
                   block=2, n_freq=5, xlim=2, skip=5, sig=False):
    """Plots in individual figures for all interactions the DMFT loops"""
    with HDFArchive(filestr.format(tp=tp, BETA=BETA), 'r') as output_files:
        urange = output_files.keys()

    for u_str in urange:
        show_conv(BETA, u_str, tp, filestr, block, n_freq, xlim, skip, sig)

        plt.show()
        plt.close()


def phase_diag(BETA, tp_range, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):

    for tp in tp_range:
        tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
        with HDFArchive(filestr.format(tp=tp, BETA=BETA), 'r') as results:
            fl_dos = []
            afm = []
            for u_str in results.keys():
                lastit = results[u_str].keys()[-1]
                labels = [name for name in results[
                    u_str][lastit]['G_iw'].indices]
                gfB_iw = results[u_str][lastit]['G_iw']
                #mesl = len(gfB_iw.mesh)/2.
                paramagnetic_hf_clean(gfB_iw, float(u_str[1:]), tp)
#
                # afmtest = np.allclose(gfB_iw['asym_up'].data[mesl:mesl+1].real,
                # gfB_iw['asym_dw'].data[mesl:mesl+1].real, 0.2)
                #afm.append(60 if afmtest else 280)

                gf_iw = np.squeeze([gfB_iw[labels[2]](i) for i in range(3)])
                fl_dos.append(gf.fit_gf(w_n[:3], gf_iw.imag)(0.))

            u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
            plt.scatter(np.ones(len(fl_dos)) * tp, u_range, c=fl_dos,
                        s=100, vmin=-2, vmax=0, cmap=plt.get_cmap('inferno'))

    plt.xlim([0, 1])
    plt.title(r'Phase diagram at $\beta={}$'.format(BETA))
    plt.xlabel(r'$t_\perp/D$')
    plt.ylabel('$U/D$')


def phase_diag_b(BETA_range, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):

    for BETA in BETA_range:
        tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=BETA))
        with HDFArchive(filestr.format(tp=tp, BETA=BETA), 'r') as results:
            fl_dos = []
            for u_str in results.keys():
                lastit = results[u_str].keys()[-1]
                labels = [name for name in results[
                    u_str][lastit]['G_iw'].indices]
                gfB_iw = results[u_str][lastit]['G_iw']
                paramagnetic_hf_clean(gfB_iw, float(u_str[1:]), tp)

                gf_iw = np.squeeze([gfB_iw[labels[2]](i) for i in range(3)])
                fl_dos.append(gf.fit_gf(w_n[:3], gf_iw.imag)(0.))

            u_range = np.array([float(u_str[1:]) for u_str in results.keys()])
            plt.scatter(u_range, np.ones(len(fl_dos)) / BETA, c=fl_dos,
                        s=150, vmin=-2, vmax=0, cmap=plt.get_cmap('inferno'))

    plt.title(r'Phase diagram at $\beta={}$'.format(BETA))
    plt.xlabel(r'$t_\perp/D$')
    plt.ylabel('$T/D$')


def averager(h5parent, h5child, last_iterations):
    """Given an H5 file parent averages over the iterations with the child"""
    sum_child = 0.
    for step in last_iterations:
        sum_child += h5parent[step][h5child]

    return 1. / len(last_iterations) * sum_child


def get_giw(h5parent, iteration_slice):
    """Recover G_iw from h5parent at iteration_slice

    Parameters
    ----------
    h5parent : h5py parent object
    iteration_slice : list or slice of iterations to average over

    Returns
    -------
    G_iw object
    """

    iterations = list(h5parent.keys())
    return averager(h5parent, 'G_iw', iterations[iteration_slice])


def get_last_table(filename, u_str, islice, name):
    g_out = []
    with tdp.HDFArchive(filename, 'r') as data:
        iterations = list(data[u_str].keys())[islice]
        for step in iterations:
            g_out.append(np.squeeze(data[u_str][step]['G_iw'][name].data))

    return np.array(g_out)


def tail_clean(gf_iw, U, tp):
    fixed = TailGf(1, 1, 3, 1)
    fixed[1] = np.array([[1]])
    fixed[2] = np.array([[-tp]])
    fixed[3] = np.array([[U**2 / 4 + tp**2 + .25]])
    gf_iw.fit_tail(fixed, 5, int(gf_iw.beta), len(gf_iw.mesh))


def paramagnetic_hf_clean(G_iw, u_int, tp):
    """Performs the average over up & dw of the green functions to
    enforce paramagnetism"""

    try:
        G_iw['asym_up'] << 0.5 * (G_iw['asym_up'] + G_iw['asym_dw'])
        tail_clean(G_iw['asym_up'], u_int, tp)

        G_iw['sym_up'] << 0.5 * (G_iw['sym_up'] + G_iw['sym_dw'])
        tail_clean(G_iw['sym_up'], u_int, -tp)

        G_iw['asym_dw'] << G_iw['asym_up']
        G_iw['sym_dw'] << G_iw['sym_up']

    except:
        G_iw['high_up'] << 0.5 * (G_iw['high_up'] + G_iw['high_dw'])
        tail_clean(G_iw['high_up'], u_int, tp)

        G_iw['low_up'] << 0.5 * (G_iw['low_up'] + G_iw['low_dw'])
        tail_clean(G_iw['low_up'], u_int, -tp)

        G_iw['high_dw'] << G_iw['high_up']
        G_iw['low_dw'] << G_iw['low_up']


def ekin(BETA, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):
    """Kinetic Energy per molecule"""
    T = []
    with HDFArchive(filestr.format(BETA=BETA, tp=tp), 'r') as results:
        for u_str in results:
            lastit = results[u_str].keys()[-3:]
            gf_iw = averager(results[u_str], 'G_iw', lastit)
            u_int = float(u_str[1:])
            paramagnetic_hf_clean(gf_iw, u_int, tp)
            hop_gf_iw = gf_iw.copy()
            for name, g0block in hop_gf_iw:
                shift = 1. if 'asym' or 'high' in name else -1
                hop_gf_iw[name] << shift * tp * gf_iw[name]

            gf_iw << hop_gf_iw + 0.25 * gf_iw * gf_iw
            T.append(gf_iw.total_density())
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(T), ur


def epot(BETA, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):
    """Potential energy per molecule"""
    V = []
    with HDFArchive(filestr.format(BETA=BETA, tp=tp), 'r') as results:
        for u_str in results:
            lastit = results[u_str].keys()[-3:]
            gf_iw = averager(results[u_str], 'G_iw', lastit)
            sig_iw = gf_iw.copy()
            u_int = float(u_str[1:])
            paramagnetic_hf_clean(gf_iw, u_int, tp)

            for name, g0block in gf_iw:
                shift = 1. if 'asym' or 'high' in name else -1
                sig_iw[name] << iOmega_n + u_int / 2. + shift * \
                    tp - 0.25 * gf_iw[name] - inverse(gf_iw[name])

            gf_iw << 0.5 * sig_iw * gf_iw
            V.append(gf_iw.total_density())
        ur = np.array([float(u_str[1:]) for u_str in results])

    return np.array(V), ur


def extract_local_sigma(BETA, tp, skip_list, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):
    """Extracts the local and intersite self-energy, averaged"""
    sd_zew, so_zew = [], []
    ur = []
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=2))
    with tdp.HDFArchive(filestr.format(BETA=BETA, tp=tp), 'r') as results:
        for u_str in results:
            if u_str in skip_list:
                continue
            gf_iw = tdp.get_giw(results[u_str], slice(-1, -3, -1))

            giw_d = .25 * (gf_iw['asym_up'] + gf_iw['sym_up'] +
                           gf_iw['asym_dw'] + gf_iw['sym_dw'])
            giw_o = .25 * \
                (-1 * gf_iw['asym_up'] + gf_iw['sym_up'] -
                 gf_iw['asym_dw'] + gf_iw['sym_dw'])
            ngiw_d = np.squeeze(giw_d.data)
            ngiw_o = np.squeeze(giw_o.data)
            ngiw_d = ngiw_d[len(ngiw_d) / 2:len(ngiw_d) / 2 + 2]
            ngiw_o = ngiw_o[len(ngiw_o) / 2:len(ngiw_o) / 2 + 2]

            siw_d, siw_o = rt.get_sigmaiw(
                1j * ngiw_d.imag, ngiw_o.real, w_n, tp)
            sd_zew.append(np.polyfit(w_n, siw_d.imag, 1))
            so_zew.append(np.polyfit(w_n, siw_o.real, 1))
            ur.append(float(u_str[1:]))

    return np.array(sd_zew), np.array(so_zew), np.array(ur)


def avg_last(BETA, tp, filestr='DIMER_PM_B{BETA}_tp{tp}.h5', over=5):
    """Averages the over the last iterations and writes inplace"""
    with tdp.HDFArchive(filestr.format(tp=tp, BETA=BETA), 'a') as output_files:
        u_range = list(output_files.keys())
        print(u_range)
        for u_str in u_range:
            giw = tdp.get_giw(output_files[u_str], slice(-over, None))
            try:
                dest_count = len(output_files[u_str])
            except KeyError:
                dest_count = 0
            dest_group = '/{}/it{:03}/'.format(u_str, dest_count)

            output_files[dest_group + 'G_iw'] = giw


def sigma_zero_freq(BETA, tp, skip_list, filestr='DIMER_PM_B{BETA}_tp{tp}.h5'):
    """Extracts the local and intersite self-energy"""
    sd_zew, so_zew = [], []
    ur = []
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=2))
    try:
        with tdp.HDFArchive(filestr.format(BETA=BETA, tp=tp), 'r') as results:
            for u_str in results:
                if u_str in skip_list:
                    continue
                gf_iw = tdp.get_giw(results[u_str], slice(-1, -3, -1))

                giw_d = .25 * (gf_iw['asym_up'] + gf_iw['sym_up'] +
                               gf_iw['asym_dw'] + gf_iw['sym_dw'])
                giw_o = .25 * \
                    (-1 * gf_iw['asym_up'] + gf_iw['sym_up'] -
                     gf_iw['asym_dw'] + gf_iw['sym_dw'])
                ngiw_d = np.squeeze(giw_d.data)
                ngiw_o = np.squeeze(giw_o.data)
                ngiw_d = ngiw_d[len(ngiw_d) / 2:len(ngiw_d) / 2 + 2]
                ngiw_o = ngiw_o[len(ngiw_o) / 2:len(ngiw_o) / 2 + 2]

                siw_d, siw_o = rt.get_sigmaiw(
                    1j * ngiw_d.imag, ngiw_o.real, w_n, tp)
                sd_zew.append(np.polyfit(w_n, siw_d.imag, 1))
                so_zew.append(np.polyfit(w_n, siw_o.real, 1))
                ur.append(float(u_str[1:]))
        return np.array(sd_zew), np.array(so_zew), np.array(ur)

    except (IndexError, RuntimeError):
        return None, None, None


def plot_zero_w_sigma(beta, tp, skip_list=[]):
    sd_zew, so_zew, ur = sigma_zero_freq(beta, tp, skip_list)
    if sd_zew is None:
        return None
    plt.figure()
    try:
        plt.plot(ur, sd_zew[:, 0], 'o', label=r'slope, $\alpha$')
        plt.plot(ur, sd_zew[:, 1], 'o', label='cut AA')
    except IndexError:
        import pdb
        pdb.set_trace()
    plt.legend(loc=0)
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r'U/D')
    plt.title(r'$\alpha$ tp{} $\beta$=100'.format(tp))
    plt.savefig('SIGMA_AA_alpha_tp{}_B100.png'.format(tp))

    plt.figure()
    plt.plot(ur, so_zew[:, 1], 's', label='cut AB')
    plt.legend(loc=0)
    plt.ylabel(r'$\Sigma_{{AB}}(w=0)$')
    plt.xlabel(r'U/D')
    plt.title(r'$\Sigma_{{AB}}(w=0)$ tp{} $\beta$=100'.format(tp))
    plt.savefig('SIGMA_AB_cut_tp{}_B100.png'.format(tp))

    plt.figure()
    plt.plot(ur, 1 / (1 - sd_zew[:, 0]), 'o', label=r'$Z$')
    plt.legend(loc=0)
    plt.ylim([0, 1])
    plt.ylabel(r'Z')
    plt.xlabel(r'U/D')
    plt.title(r'$Z$ tp{} $\beta$=100'.format(tp))
    plt.savefig('Z_tp{}_B100.png'.format(tp))

    plt.figure()
    plt.plot(ur, (tp + so_zew[:, 1]) / (1 - sd_zew[:, 0]), 's', label='cut AB')
    plt.legend(loc=0)
    plt.ylabel(r'$Z(t_\perp + \Sigma_{{AB}}(w=0))$')
    plt.xlabel(r'U/D')
    plt.title(r'$Z(t_\perp + \Sigma_{{AB}}(w=0))$ tp{} $\beta$=100'.format(tp))
    plt.savefig('ZSIGMA_AB_cut_tp{}_B100.png'.format(tp))


def extract_density_correlators(filename, skiplist):
    """Recover from file the measured density correlators"""
    with HDFArchive(filename, 'r') as datarecord:
        nn = []
        n = []
        u = []
        for uk in datarecord:
            if uk in skiplist:
                continue
            last_it = list(datarecord[uk].keys())[-1]
            nn.append(datarecord[uk][last_it]['density'])
            n.append(datarecord[uk][last_it]['occup'])
            u.append(float(uk[1:]))
    nn = np.array(nn)
    n = np.array(n)
    u = np.array(u)
    return nn, n, u


def plot_cor(ldensity_cor, u_int):
    """plot Density correlators"""
    fig, axe = plt.subplots(3, 1, sharex=True)
    d = np.mean(ldensity_cor[:, [1, 4]], axis=1)
    axe[0].plot(u_int, d, '+:')
    axe[0].set_ylabel(r'$\langle d \rangle$')
    mfl = 1 - 2 * d
    axe[1].plot(u_int, mfl, 'x:')
    axe[1].set_ylabel(r'$\langle (n_\uparrow - n_\downarrow)^2 \rangle$')
    msc = ldensity_cor.T[0] - ldensity_cor.T[2] - \
        ldensity_cor.T[3] + ldensity_cor.T[5]
    axe[2].plot(u_int, msc, 'x:')
    axe[2].set_ylabel(r'$\langle m_A m_B \rangle$')
    axe[2].set_xlabel('U/D')
    fig.subplots_adjust(hspace=0)
