#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Extract the double occupation from dimer data files
===================================================

"""
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import argparse
import re
from glob import glob
import numpy as np
import py3qs.triqs_dimer as tdm


def density_obs(moments):
    moments = np.array(moments).T
    d = (moments[1] + moments[4]) / 2
    ma = (moments[0] - moments[2] - moments[3] + moments[5])
    return d, ma


def extract_obs(filelist, u_shift=0):
    u_list = []
    tp_list = []
    d_list = []
    for filename in filelist:
        tp = re.findall(r'PM_.+tp(\d+\.\d+)', filename)[0]
        try:
            nn, u = tdm.extract_density_correlators(filename, 'density')
            d = density_obs(nn)[0]
            u_list.append(u + u_shift)
            tp_list.append(float(tp))
            d_list.append(d)
        except IOError:
            pass
    return u_list, tp_list, d_list


def main():
    parser = argparse.ArgumentParser(description="Extract and save the double occupation of all datafiles to construct a phase diagram",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-B', '--beta', type=float, default=100.0,
                        help="Simulation inverse temperature")
    parser.add_argument('workdir', help="Data directory")

    args = parser.parse_args()
    beta = args.beta
    workdir = args.workdir

    # metal seeds
    datafiles = glob(workdir + '/DIMER_PM*_B{}_tp*h5'.format(beta))
    mu_list, mtp_list, md_list = extract_obs(datafiles)
    np.savez('met_B{}_Utp'.format(beta), u_int=mu_list,
             tp=mtp_list, docc=md_list)
    print("Saved data for metallic seed files for Beta{}".format(beta))

    # insulator seeds
    datafiles = glob(workdir + '/DIMER_PM_ins*_B{}_tp*h5'.format(beta))
    iu_list, itp_list, id_list = extract_obs(datafiles, 1e-5)
    np.savez('ins_B{}_Utp'.format(beta), u_int=iu_list,
             tp=itp_list, docc=id_list)
    print("Saved data for insulating seed files for Beta{}".format(beta))


if __name__ == '__main__':
    main()
