# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Installing packages on code for DMFT
"""
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

import dmft
import sys
import numpy as np

class PyTest(TestCommand):
    """Test class to do test coverage analysis"""
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--cov-report', 'term-missing',
                          '--cov', 'dmft', 'tests/']
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
    name="dmft-learn",
    description="Educative code on DMFT",
    version=dmft.__version__,
    packages=find_packages(),
    author="Óscar Nájera",
    author_email='najera.oscar@gmail.com',
    license="GNU General Public License v3 (GPLv3)",

    install_requires=['numpy', 'scipy', 'matplotlib', 'slaveparticles'],
    setup_requires=['Sphinx', 'cython'],
    tests_require=['pytest', 'pytest-cov'],
    cmdclass={'test': PyTest, 'build_ext': build_ext},
    ext_modules =[Extension('hffast', ['dmft/hirschfye_cy.pyx'],
                            include_dirs=[np.get_include()],
                            libraries=['cblas', 'gsl']),
                  ],
)
