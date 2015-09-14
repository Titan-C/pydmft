
from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq
from pytriqs.plot.mpl_interface import oplot
from pytriqs.archive import HDFArchive
import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22})


def show_conv(beta, u_int, filestr='B{}_U{}.h5', block=2, n_freq=5, xlim=2):
    """Plot the evolution of the Green's function in DMFT iterations"""
    _, axes = plt.subplots(1, 2, figsize=(13, 8))
    freq_arr = []
    with HDFArchive(filestr.format(beta), 'r') as datarecord:
        for step in datarecord:
            labels = [name for name in datarecord[step]['G_iw'].indices]
            gf_iw = datarecord[step]['G_iw'][labels[block]]
            axes[0].oplot(gf_iw, 'o:')
            freq_arr.append(gf_iw.data[:n_freq, 0, 0].imag)
    freq_arr = np.asarray(freq_arr).T
    for num, freqs in enumerate(freq_arr):
        axes[1].plot(freqs, 'o-.', label=str(num))
    axes[0].set_xlim([0, xlim])
    axes[1].legend(loc=0, ncol=n_freq)
    graf = r'$G(i\omega_n)$'
    axes[0].set_title(r'Change of {} @ $\beta={}$, U={}'.format(graf, beta, u_int))
    axes[0].set_ylabel(graf)
    axes[0].set_xlabel(r'$i\omega_n$')
    axes[1].set_title('Evolution of the first frequencies')
    axes[1].set_ylabel(graf+'$(l)$')
    axes[1].set_xlabel('iterations')
