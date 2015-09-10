# -*- coding: utf-8 -*-
"""
Analyzing the single band single site Metal-Insulator Transition
================================================================

Reviewing the single band case and expecting to find the famous
coexistence region I start looping at many different points in the
phase diagram. The iterate.py script is this same folder is reponsible
for creating the data but it still requires manual input for the
searched data point.
"""

import dmft.plot.cthyb_h_single_site as plot

###############################################################################
# I start first by checking the convergence of the system at various
# data points for this I have a look at the evolulution of the
# outputed Green's functions and the self energy on every loop
# performing as many iteration to see the system stabilize

plot.show_conv(64., 2.4, 'coex/B{}_U{}/Sig.out.*')

###############################################################################
# This first plot demostrates that for the simply metalic state the
# system is quite well behaved and the convergence is quite
# simple. Few iterations are necessary but then there always remains
# the monte carlo noise in the solution.
