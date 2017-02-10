from __future__ import division, absolute_import, print_function

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re
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

datafiles = glob('DIMER_PM_met*_B*h5')
datafiles += glob('DIMER_PM_ins*_B*h5')

for filename in datafiles:
    state, beta = re.findall(r'PM_(...)_B(\d+\.\d+)', filename)[0]
    try:
        nn, u = tdm.extract_density_correlators(filename, 'density')
        d, ma = density_obs(nn)
        met_sh = 60 if state == 'met' else 20
        # if state == 'ins':
        #    continue
        plt.scatter(u, np.ones_like(u) / float(beta), c=d, s=met_sh,
                    vmin=0.04, vmax=0.13, cmap=plt.get_cmap('viridis'))
    except IOError:
        pass
        # plt.legend(loc=0)
plt.xlim([1.5, 3])
plt.ylim([0, 0.04])

boundaries = np.array([(1.91, 300.), (1.91, 200.), (1.93, 100.), (1.99, 64.), (2.1, 44.23),
                       (2.14, 41.56), (2.18, 40.), (2.18, 64.), (2.18, 100.), (2.19, 200.), (2.21, 300.)]).T
plt.plot(boundaries[0], 1 / boundaries[1], '+-')
plt.xlabel(r'$U/D$')
plt.ylabel(r'$T/D$')
plt.savefig('phasediag.png')
plt.show()
