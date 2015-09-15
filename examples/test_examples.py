# -*- coding: utf-8 -*-
"""testing the examples before gallery"""
from __future__ import division, print_function, absolute_import
import dmft.plot.hf_single_site as pss
import matplotlib
import subprocess
matplotlib.use('Agg')


def test_example():
    """Testing a very fast single site execution of HF"""
    command = "examples/Hirsh-Fye/single_site.py -sweeps 1000 -therm 400 -Niter 6 -ofile /tmp/testhfss"
    command = command.split()
    print(subprocess.call(command))
    pss.show_conv(4, 'U2.5', '/tmp/testhfss', xlim=8)
