#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Installing packages on code for DMFT
"""
from setuptools import setup, find_packages
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

import dmft
import numpy as np

with open('README.rst') as f:
    long_description = f.read()

setup(
    name="dmft-learn",
    description="Educative code on DMFT",
    long_description=long_description,
    version=dmft.__version__,
    packages=find_packages(),
    author="Óscar Nájera",
    author_email='najera.oscar@gmail.com',
    license="GNU General Public License v3 (GPLv3)",

    install_requires=['numpy', 'scipy', 'matplotlib', 'slaveparticles',
                      'joblib', 'pandas', 'numba', 'h5py', 'mpi4py'],
    setup_requires=['sphinx', 'cython', 'pytest-runner'],
    tests_require=['pytest-cov', 'pytest'],  # Somehow this order is relevant
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('dmft.hffast', ['dmft/hirschfye_cy.pyx',
                                           'dmft/hfc.cpp'],
                           include_dirs=[np.get_include()],
                           language="c++",
                           extra_compile_args=["-std=c++11"],
                           libraries=['gsl', 'openblas']),
                 ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ]
)
