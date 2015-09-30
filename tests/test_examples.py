# -*- coding: utf-8 -*-
"""
Testing the examples before gallery
======================================

Performs a quick run of the examples to keep tests upon them
"""

from __future__ import division, print_function, absolute_import
import subprocess
# matplotlib back end has to be called before it gets loaded elsewhere
import matplotlib
matplotlib.use('Agg')
import dmft.plot.hf_single_site as pss


def test_example():
    """Testing a very fast single site execution of HF"""
    command = "examples/Hirsh-Fye/single_site.py -sweeps 100 -therm 400 -Niter 4 -ofile /tmp/testhfss"
    command = command.split()
    assert not subprocess.call(command)
    pss.show_conv(4, 'U2.5', '/tmp/testhfss', xlim=8)

    command += '-new_seed 2.5 2.8 4'.split()
    print(command)
    assert not subprocess.call(command)
    pss.show_conv(4, 'U2.8', '/tmp/testhfss', xlim=8)
