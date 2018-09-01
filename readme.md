# PyDMFT

A python library to work on quantum interacting systems. It is focused for
research using the Dynamical Mean Field Theory (DMFT)[dmft-rmp](#dmft-rmp)

The particular highlight of this library is the gallery of examples for
which I have developed the scripts to reproduce known results in this field
and also provide with my personal research work, showing how it was done
and referencing the publication it appeared. Other examples are just test
that resulted in dead ends. Examples exist but not all of them are well
documented.

You can access the documentation in (https://titan-c.github.io/pydmft)

If you find problems with it, open an issue or even better a pull
request. If you use it cite it.

This library was developed during my Doctoral research stay at the
[Laboratoire de Physique des solides in Orsay, France](https://www.lps.u-psud.fr/?lang=fr).

## Similar Projects

There are other projects to work on this. PyDMFT is just mine and focuses
on using python and providing examples.

- [TRIQS](https://triqs.ipht.cnrs.fr)
- [ALPSCore](http://alpscore.org/)

# Installation

This package is python2 and python3 compatible, but let's try go be in the
present and use python3 unless you have really ancient dependencies.

I also prefer working in linux systems and thus the instructions are suited
for them.

## Virtual environment

Although you can install this package system-wide I recommend using a
python virtual environment. There are few options, to do this search on the
internet to learn more and stay up to day on the current trends. For the
moment, and specially for scientific python I use the
[conda](https://conda.io/docs/index.html) virtual environments.

The next commands for your terminal help you install conda and setup a
virtual environment named `pydmft` with the dependencies for this project.

```bash
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=$HOME/miniconda3/bin:$PATH
conda update --yes --quiet conda
# Install python 3.6 and dependencies in the pydmft virtual environment
conda create --yes --quiet -n pydmft python=3.6 nomkl openblas pip scipy numpy matplotlib sphinx setuptools cython tornado pyzmq h5py joblib pandas numba
source activate testenv
pip install slaveparticles sphinx-gallery mako mpi4py cmocean
```

## Installing the package

This clones the repository and let's you install it on development
version. Which means that if you edit the code those changes will affect
your installed version. If that is not a behavior you desire change
`develop` for `install` in the last line.

```bash
git clone https://github.com/Titan-C/pydmft.git
cd pydmft
python setup.py develop
```


# References

<b id="dmft-rmp">dmft-rmp</b> Georges, A., Kotliar, G., Krauth, W., & Rozenberg, M. J.,
Dynamical mean-field theory of strongly correlated fermion systems and the
limit of infinite dimensions, Reviews of Modern Physics, 68(1), 13–125
(1996).  http://dx.doi.org/10.1103/revmodphys.68.13
