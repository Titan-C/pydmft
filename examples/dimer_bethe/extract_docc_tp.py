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
    temp_list = []
    d_list = []
    for filename in filelist:
        beta = re.findall(r'PM_..._B(\d+\.\d+)', filename)[0]
        try:
            nn, u = tdm.extract_density_correlators(filename, 'density')
            d = density_obs(nn)[0]
            u_list.append(u + u_shift)
            temp_list.append(np.ones_like(u) / float(beta))
            d_list.append(d)
        except IOError:
            pass
    return u_list, temp_list, d_list


def main():
    parser = argparse.ArgumentParser(description="Extract and save the double occupation of all datafiles to construct a phase diagram",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-tp', type=float, default=0.3,
                        help="Dimer hybridization strength")
    parser.add_argument('workdir', help="Data directory")

    args = parser.parse_args()
    tp = args.tp
    workdir = args.workdir

    # metal seeds
    datafiles = glob(workdir + '/DIMER_PM_met*_B*_tp{}*h5'.format(tp))
    mu_list, mtemp_list, md_list = extract_obs(datafiles)
    np.savez('met_tp{}_UTd'.format(tp), u_int=mu_list,
             temp=mtemp_list, docc=md_list)
    print("Saved data for metallic seed files for tp{}".format(tp))

    # insulator seeds
    datafiles = glob(workdir + '/DIMER_PM_ins*_B*_tp{}*h5'.format(tp))
    iu_list, itemp_list, id_list = extract_obs(datafiles, 1e-5)
    np.savez('ins_tp{}_UTd'.format(tp), u_int=iu_list,
             temp=itemp_list, docc=id_list)
    print("Saved data for insulating seed files for tp{}".format(tp))

if __name__ == '__main__':
    main()
