# -*- coding: utf-8 -*-
r"""
H5PY interface
==============
"""
# Created Thu Oct 15 19:06:28 2015
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import json
import os
from h5py import File
import numpy as np


def add_attributes(parent, attr):
    """
    add attributes to an h5py data item

    :param obj parent: h5py parent object
    :param dict attr: dictionary of attributes
    """
    if isinstance(attr, dict):
        # attr is a dictionary of attributes
        for key, item in attr.items():
            parent.attrs[key] = item


def get_attributes(parent):
    """
    get the attributes from the h5py data item

    :param obj parent: h5py parent object
    """
    return dict(parent.attrs.items())


def _make_npy(name, obj):
    try:
        data = obj.value
        np.save(name, data)
    except AttributeError:
        if not os.path.exists(name):
            os.makedirs(name)
        setup = get_attributes(obj)
        setup = {str(a): v for a, v in setup.items()
                 if not isinstance(v, np.bool_)}
        if setup:
            with open(os.path.dirname(name) + '/setup', 'w') as fname:
                json.dump(setup, fname, indent=2,
                          default=lambda x: x.decode() if isinstance(x, bytes) else int(x))


def h5_2_npy(h5file):
    """Converts a h5 data file into a folder tree with npy files"""
    target_dir = os.path.splitext(h5file)[0]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    cwd = os.getcwd()
    try:
        with File(h5file, 'r') as source:
            os.chdir(target_dir)
            source.visititems(_make_npy)
    finally:
        os.chdir(cwd)


def make_gf(h5data):
    """Take Green function data from h5 group and complete negative frequencies"""
    pos_giw = np.squeeze(h5data.view(np.complex128))
    return np.concatenate((pos_giw[::-1].conjugate(), pos_giw))


def dimer_gf(filename, u_int, iteration):
    """Extract specific iteration greenfunction of h5 dimer data

    Dimer greenF has 4 blocks 'sym_up', 'sym_dw', 'asym_up',
    'asym_dw', output is in that order.

    example: giw=dimer_gf("DIMER.h5", "U2.15", -1)
    """

    with File(filename, 'r') as datarecord:
        iteration = list(datarecord[u_int])[iteration]
        dat = [make_gf(datarecord[u_int][iteration]['G_iw'][name]['data'].value)
               for name in ['sym_up', 'sym_dw', 'asym_up', 'asym_dw']]
    return np.array(dat)
