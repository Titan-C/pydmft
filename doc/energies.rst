===============
Energy formulas
===============

.. _potential_energy:

Potential energy
================

According to [Fetter_Walecka]_ in equation 23.14 then transformed to Matsubara frequencies the potential energy can be described by:

.. math:: \langle V \rangle = \frac{1}{\beta} \sum_{k,n} \frac{1}{2}\left(
   i\omega_n - \epsilon_k^0 \right)Tr G(k, i\omega_n)

And expressing it in local quantities with the DMFT approximation that the Self-Energy is local


.. math:: \langle V \rangle = \frac{1}{\beta} \sum_{n} \frac{1}{2}
    Tr(\Sigma(i\omega_n)G(i\omega_n))
    :label: local_potential_energy

.. [Fetter_Walecka] Fetter, Walecka, Quantum Theory of many-particle systems
