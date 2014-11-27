====================================
Two site Dynamical Mean-Field Theory
====================================

The DMFT is a valuable method to investigate the physics of strongly
correlated electrons on a lattice. It consists of a procedure to map
the original lattice problem into an effective impurity model which
describes a single correlate impurity orbital embedded in an uncorrelated
bath of conduction-band states. This mapping is designed to be
self-consistent, that is the bath parameters depend on the on-site lattice
Green's function.

The impurity model is the critical problem in the DMFT since it poses a highly
non trivial many-body problem that must be solved repeatedly.

There is no problem to treat the single-impurity Anderson model with a small
number of lattice sites :math:`n_s` numerically exact. But for a finite number of
sites, the self-consistent mapping between the Hubbard model into the impurity
model is approximate. The exact solution of the effective impurity model is
thus achieved at the expense of an approximate self-consistency.

Single band Hubbard model in the Bethe lattice
==============================================

The single band Hubbard Hamiltonian reads:

.. math::
   \mathcal{H} = - t \sum_{<i,j>, \sigma} (c^\dagger_{i\sigma}c_{j\sigma} +h.c.)
    + U  \sum_{i\sigma} n_{i\uparrow}n_{i\downarrow}

The is hopping only between nearest neighbors. For a paramagnetic, spatially
homogeneous phase the on-site Green Function :math:`G(\omega) = \braket{c_{i\sigma}
c_{i\sigma^\dagger}}` is give by

.. math::
   G(\omega) = \int_{-\infty}^{\infty} dx \frac{\rho_0(x)}{\omega + \mu - x - \Sigma(\omega)}

In this case of the Bethe lattice, the self-energy is local in the limit of
infinite coordination. The free density of states is

.. math:: \rho_0(x) = \frac{\sqrt{4t^2 - x^2}}{2\pi t^2}

Single impurity Anderson model(SIAM)
====================================

The Hamiltonian reads:

.. math:: (\epsilon_d -\mu) d^\dagger_\sigma d_sigma +
   U d^\dagger_\uparrow d_\uparrow d^\dagger_\downarrow d_\downarrow
   \sum_{\sigma,k} (\epsilon_k - \mu) a^\dagger_{k\sigma}a_{k\sigma}
   + \sum_{\sigma,k} V_k(d^\dagger_\sigma c_{k\sigma} + h.c.)

The impurity Green function $G(\omega) = \braket{d_{\sigma}d_{\sigma^\dagger}}$
is given by

.. math:: G_{imp}(\omega) = [ \omega + \mu - \epsilon_d \Delta(\omega) - \Sigma_{imp}(\omega) ]^{-1}

where the hybridization function is :math:`\Delta(\omega) = \sum_k |V_k|^2 /
(\omega + \mu - \epsilon_k)`. The DMFT self-consistency requires that

.. math:: G_{imp}(\omega) = G(\omega)
   :label: DMFT_selfconsistency

as the self-energy is the same

.. math:: \Sigma_{imp}(\omega) = \Sigma(\omega)

The two site single impurity model
----------------------------------

The self-consistency condition :eq:`DMFT_selfconsistency` can be fulfilled only
for :math:`n_s \rightarrow \infty` that is a bath with an infinite number of degrees
of freedom. This poses the SIAM as a many-body problem. The simplification now
is to contruct a two site DMFT, where there is only an impurity and one bath site.

Thus the hybridization function reduces to :math:`\Delta(\omega)=V^2/(\omega+ \mu -\epsilon_c)`

and the free :math:`(U=0)` impurity Green function is a two pole function

.. math::
   G^{(0)}_{imp} =& \frac{\omega - \epsilon_c + \mu}{(\omega + \mu - \epsilon_d)(\omega + \mu \epsilon_c) - V^2} \\
    =& \frac{1}{2r} \left( \frac{r-\delta\epsilon}{\omega + \mu - \bar{\epsilon} +r}
    + \frac{r + \delta\epsilon}{\omega + \mu - \bar{\epsilon} - r} \right)

where :math:`\bar{\epsilon}=(\epsilon_d+\epsilon_c)/2`, :math:`\delta{\epsilon}
=(\epsilon_d-\epsilon_c)/2` and :math:`r=\sqrt{\delta \epsilon^2 + V^2}`. The
interacting Green function has four poles and the self-energy two poles.
