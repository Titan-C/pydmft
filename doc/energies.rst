===============
Energy formulas
===============

.. _kinetic_energy:

Kinetic Energy
==============

This energy is the weigthed average of non-interacting kinetic
energy. Assuming a non-interacting Hamiltonian in momentum space the
kinetic average is.

.. math:: \langle T \rangle  = Tr \frac{1}{\beta} \sum_{k,n} \epsilon_k^0 G(k, i\omega_n)

It can be transformed into a treatable form relying on local quantities

.. math:: \langle T \rangle  = Tr \frac{1}{\beta} \sum_{k,n} \left( \epsilon_k^0 G(k, i\omega_n) + G(k, i\omega_n)^{-1}G(k, i\omega_n) - G^{free}(k, i\omega_n)^{-1}G^{free}(k, i\omega_n) \right)

.. math::  = Tr \frac{1}{\beta} \sum_{k,n} \left( \epsilon_k^0 G(k, i\omega_n) + (i\omega_n - \epsilon_k^0 - \Sigma(i\omega_n))G(k, i\omega_n) - (i\omega_n - \epsilon_k^0)G^{free}(k, i\omega_n) \right)

.. math::  = Tr \frac{1}{\beta} \sum_{k,n} \left( i\omega_n \left( G(k, i\omega_n)- G(k, i\omega_n)^{free} \right) - \Sigma(i\omega_n) G(k, i\omega_n) + \epsilon_k^0G^{free}(k, i\omega_n) \right)

The first two terms can be summed in reciprocal space to yield a
local the quantities that come out of the DMFT self-consistency and
the last term as it belongs to the non-interacting system is
trivially solvable

.. math::  \langle T \rangle = Tr \frac{1}{\beta} \sum_n \left( i\omega_n \left( G(i\omega_n)- G(i\omega_n)^{free} \right) - \Sigma(i\omega_n)G(i\omega_n) \right) + \int_{-\infty}^\infty \epsilon\rho(\epsilon)n_F(\epsilon-\mu) d\epsilon

It is also possible to take a simpler approac by introducing a zero to
the frequecy sum, with a constant factor. In this case one takes from

.. math:: \langle T \rangle  = Tr \frac{1}{\beta} \sum_{k,n} \left( \epsilon_k^0 G(k, i\omega_n) + G(k, i\omega_n)^{-1}G(k, i\omega_n) \right)

.. math::  = Tr \frac{1}{\beta} \sum_{k,n} \left( \epsilon_k^0 G(k, i\omega_n) + (i\omega_n - \epsilon_k^0 - \Sigma(i\omega_n))G(k, i\omega_n) \right)

But the local self-energy can be expresed by

.. math:: \Sigma(i\omega_n) = \mathcal{G}^{0, -1} - G(i\omega_n)^{-1} =  i\omega_n - h_{loc} - \Delta(i\omega_n) - G(i\omega_n)^{-1}

Where :math:`h_{loc}` is the momentum independent part of the
hamiltonian. Then the expression transforms into.

.. math:: \langle T \rangle = Tr \frac{1}{\beta} \sum_{k,n} \left(h_{loc} + \Delta(i\omega_n)\right) G(k, i\omega_n) = Tr \frac{1}{\beta} \sum_n \left(h_{loc} + \Delta(i\omega_n)\right) G(i\omega_n)


.. _potential_energy:

Potential energy
================

According to [Fetter-Walecka]_ in equation 23.14 then transformed to Matsubara frequencies the potential energy can be described by:

.. math:: \langle V \rangle = \frac{1}{\beta} \sum_{k,n} \frac{1}{2}\left(
   i\omega_n - \epsilon_k^0 \right)Tr G(k, i\omega_n)

And expressing it in local quantities with the DMFT approximation that the Self-Energy is local


.. math:: \langle V \rangle = \frac{1}{\beta} \sum_{n} \frac{1}{2}
    Tr(\Sigma(i\omega_n)G(i\omega_n))
    :label: local_potential_energy

References
----------

.. [Fetter-Walecka] Fetter, Walecka, Quantum Theory of many-particle systems
