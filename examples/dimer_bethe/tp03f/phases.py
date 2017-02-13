from __future__ import division, absolute_import, print_function

import re
from glob import glob
import numpy as np
import numpy.ma as ma
import matplotlib
from matplotlib.mlab import griddata
matplotlib.use('agg')
import matplotlib.pyplot as plt
import py3qs.triqs_dimer as tdm

plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22,
                                'mathtext.fontset': 'cm'})


def density_obs(moments):
    moments = np.array(moments).T
    d = (moments[1] + moments[4]) / 2
    ma = (moments[0] - moments[2] - moments[3] + moments[5])
    return d, ma


workdir = "/home/oscar/orlando/dev/dmft-learn/examples/dimer_bethe/tp03f/"
workdir = "/scratch/oscar/dimer_bethe/tp03f/"
workdir = ""

datafiles = glob(workdir + 'DIMER_PM_met*_B*h5')

mu_list = []
mT_list = []
md_list = []
for filename in datafiles:
    state, beta = re.findall(r'PM_(...)_B(\d+\.\d+)', filename)[0]
    try:
        nn, u = tdm.extract_density_correlators(filename, 'density')
        d = density_obs(nn)[0]
        mu_list.append(u)
        mT_list.append(np.ones_like(u) / float(beta))
        md_list.append(d)
    except IOError:
        pass

datafiles = glob(workdir + 'DIMER_PM_ins*_B*h5')

iu_list = []
iT_list = []
id_list = []
for filename in datafiles:
    state, beta = re.findall(r'PM_(...)_B(\d+\.\d+)', filename)[0]
    try:
        nn, u = tdm.extract_density_correlators(filename, 'density')
        d = density_obs(nn)[0]
        iu_list.append(u + 1e-5)
        iT_list.append(np.ones_like(u) / float(beta))
        id_list.append(d)
    except IOError:
        pass

# Join double occupation results
x = np.concatenate(mu_list + iu_list)
y = np.concatenate(mT_list + iT_list)
z = np.concatenate(md_list + id_list)
# define plot grid.
xi = np.linspace(1.7, 2.5, 150)
yi = np.linspace(0, 0.04, 150)
# Grid the data
zi = griddata(x, y, z, xi, yi, interp='nn')
CS = plt.contourf(xi, yi, zi, 15, cmap=plt.get_cmap('viridis'))
plt.colorbar()  # draw colorbar

# plot data points.
boundaries = np.array([(1.91, 1e5), (1.91, 300.), (1.91, 200.), (1.93, 100.),
                       (1.99, 64.), (2.115, 44.23), (2.145, 41.56),
                       (2.18, 40.), (2.18, 64.), (2.18, 100.), (2.19, 200.),
                       (2.205, 300.), (2.24, 1e5)]).T
DH0 = np.array([(2.05, 1e5), (2.05, 300.), (2.05, 200.), (2.05, 100.),
                (2.05, 64.), (2.07, 50.732), (2.12, 44.23), (2.18, 40.)]).T

plt.plot(DH0[0], 1 / DH0[1], 'rx-', lw=3)
plt.fill(boundaries[0], 1 / boundaries[1], 'k+-', alpha=0.5, lw=4)

plt.scatter(np.concatenate(mu_list), np.concatenate(mT_list),
            c=np.concatenate(md_list), s=70, vmin=0.036, vmax=0.12,
            cmap=plt.get_cmap('viridis'), marker='o', edgecolor='k')

plt.scatter(np.concatenate(iu_list), np.concatenate(iT_list),
            c=np.concatenate(id_list), s=30, vmin=0.036, vmax=0.12,
            cmap=plt.get_cmap('viridis'), marker='o', edgecolor='k')

plt.xlim([1.7, 2.5])
plt.ylim([0, 0.04])

plt.xlabel(r'$U/D$')
plt.ylabel(r'$T/D$')
plt.savefig('phasediag.png')
