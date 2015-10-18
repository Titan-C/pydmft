# -*- coding: utf-8 -*-
"""
Testing the examples before gallery
======================================

Performs a quick run of the examples to keep tests upon them
"""

from __future__ import division, print_function, absolute_import
import subprocess
import pytest
# matplotlib back end has to be called before it gets loaded elsewhere
import matplotlib
matplotlib.use('Agg')
import dmft.plot.hf_single_site as pss
import dmft.plot.hf_dimer as pdp


@pytest.mark.parametrize("case, ofile, plot",
                         [('single_site', 'testhfss', pss),
                          ('dimer_pm', 'testdp', pdp)])
def test_example(case, ofile, plot):
    """Testing a very fast HF runs"""
    command = "examples/Hirsh-Fye/{}.py -sweeps 100 -therm 400 -Niter 2 -ofile /tmp/{}".format(case, ofile)
    command = command.split()
    assert not subprocess.call(command)
    plot.show_conv(4, 'U2.5', '/tmp/'+ofile, xlim=8)

    command += '-new_seed 2.5 2.8 4'.split()
    print(command)
    assert not subprocess.call(command)
    plot.show_conv(4, 'U2.8', '/tmp/'+ofile, xlim=8)
