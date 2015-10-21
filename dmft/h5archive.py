# -*- coding: utf-8 -*-
r"""
H5PY interface
==============
"""
# Created Thu Oct 15 19:06:28 2015
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
from h5py import File


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
