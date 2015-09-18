
from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq
from pytriqs.plot.mpl_interface import oplot
from pytriqs.archive import HDFArchive
from dmft.plot.hf_single_site import label_convergence
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_str, filestr='B{}_U{}.h5', block=2, n_freq=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8))
    freq_arr = []
    with HDFArchive(filestr.format(beta), 'r') as datarecord:
        for step in datarecord[u_str]:
            labels = [name for name in datarecord[u_str][step]['G_iw'].indices]
            gf_iw = datarecord[u_str][step]['G_iw'][labels[block]]
            axes[0].oplot(gf_iw, 'o:', label=None)
            freq_arr.append(gf_iw.data[:n_freq, 0, 0].imag)
    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        axes[1].plot(freqs, 'o-.', label=str(num))

    graf = r'$G(i\omega_n)$'
    label_convergence(beta, u_str, axes, graf, n_freq, xlim)
