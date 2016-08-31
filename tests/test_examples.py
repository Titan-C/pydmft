# -*- coding: utf-8 -*-
"""
Testing the examples before gallery
======================================

Performs a quick run of the examples to keep tests upon them
"""

from __future__ import division, print_function, absolute_import
import subprocess
import os
import sys
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
    command = "examples/Hirsh-Fye/{}.py -sweeps 100 -therm 400 -Niter 2 -ofile /tmp/{}".format(
        case, ofile)
    command = command.split()
    assert not subprocess.call(command)
    plot.show_conv(4, 2.5, filestr='/tmp/' + ofile, xlim=8, skip=0)

#
#skip_list = ['energies', 'stability', 'scattering']
# example_files = os.listdir(
    #'examples') + os.listdir('examples/IPT') + os.listdir('examples/RKKY')
# plot_list = [pfl for pfl in example_files if pfl.startswith(
    #'plot') and pfl.endswith('.py') and not any(x in pfl for x in skip_list)]
# sys.path.append('examples')
# sys.path.append('examples/IPT')
# sys.path.append('examples/RKKY')
#
#
#@pytest.mark.parametrize("plot", plot_list)
# def test_plots(plot):
    #exec('import ' + plot[:-3])
