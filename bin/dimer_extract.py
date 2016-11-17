# -*- coding: utf-8 -*-
r"""
Extracts the Green functions from HDF5 files
"""
# Created Fri Oct 28 15:30:18 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import argparse
import os
import numpy as np

from dmft.plot.triqs_dimer import extract_flat_gf_iter, list_availible_data

parser = argparse.ArgumentParser(description='Extracts data from dimer HDF5 files',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('filename', help='HDF5 data file')
parser.add_argument('U', type=float, help='Local interaction value')
parser.add_argument('last', type=int, help='Count from last iterations')
parser.add_argument('-l', '--list', action='store_true',
                    help='Lists the available data')

args = parser.parse_args()
if args.list:
    list_availible_data(args.filename, args.U, args.last)

gfs = extract_flat_gf_iter(args.filename, args.U, args.last)
np.save(os.path.splitext(args.filename)[0] + 'U' + str(args.U), gfs)
