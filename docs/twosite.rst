====================================
Two site Dynamical Mean-Field Theory
====================================

The Dynamical mean field theory has become a valuable method to investigate the physics
of strongly correlated electrons in a lattice. It consist of a procedure in
which the original lattice problem is mapped into an effective impurity model
that describes a single correlated impurity orbital embedded in an uncorrelated
bath of conduction-band states. This mapping is a self-consistent one, where
the bath parameters depend on the on-site impurity Green's function.

The impurity model is crucial in the DMFT as it poses a highly non-trivial
many-body problem that must be solved repeatedly, incrementing the cost
of solving the problem computationally and limiting the applicability of
the method.

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
   G(\omega) = \int_{-\infty}^{\infty} dx
   \frac{\rho_0(x)}{\omega + \mu - x - \Sigma(\omega)}
   :label: Site_GF_H_Trans

In this case of the Bethe lattice, the self-energy is local in the limit of
infinite coordination. The free density of states is

.. math:: \rho_0(x) = \frac{\sqrt{4t^2 - x^2}}{2\pi t^2}
   :label: Bethe_dos

Single impurity Anderson model(SIAM)
====================================

The Hamiltonian reads:

.. math:: \mathcal{H} =&\sum_\sigma(\epsilon_d -\mu) d^\dagger_\sigma d_\sigma +
   U d^\dagger_\uparrow d_\uparrow d^\dagger_\downarrow d_\downarrow \\
   & + \sum_{\sigma,k} (\epsilon_k - \mu) a^\dagger_{k\sigma}a_{k\sigma}
   + \sum_{\sigma,k} V_k(d^\dagger_\sigma c_{k\sigma} + h.c.)
   :label: SIAM

The impurity Green function :math:`G_{imp}(\omega) = \braket{d_{\sigma}d_{\sigma^\dagger}}`
is given by

.. math:: G_{imp}(\omega) =
    [ \omega + \mu - \epsilon_d - \Delta(\omega) - \Sigma_{imp}(\omega) ]^{-1}
    :label: ImpGF_gen

where the hybridization function is :math:`\Delta(\omega) = \sum_k |V_k|^2 /
(\omega + \mu - \epsilon_k)`.

The DMFT self-consistency requires that

.. math:: G_{imp}(\omega) = G(\omega)
   :label: DMFT_selfconsistency

as the self-energy is required to be the same

.. math:: \Sigma_{imp}(\omega) = \Sigma(\omega)

In this fashion the original lattice problem is mapped onto the SIAM and can be
solved by the iterative process that: Starts with a Guess of the Self-Energy,
that allows to calculate through equation :eq:`Site_GF_H_Trans` the on-site
lattice Greens Function. The DMFT Self-consistency :eq:`DMFT_selfconsistency`
allows to use this Green function to determine a new hybridization function
:math:`\Delta(\omega)` and thus the parameters of the effective SIAM. Then the
impurity problem is solved to get a new estimate of the impurity self-energy that
is used into a new cycle until self-consistency is achieved.

The two site single impurity model
----------------------------------

The self-consistency condition :eq:`DMFT_selfconsistency` can be fulfilled only
for :math:`n_s \rightarrow \infty` that is a bath with an infinite number of degrees
of freedom. This poses the SIAM as a many-body problem. The simplification now
is to construct a two site DMFT, where there is only an impurity and one bath site.
As such the only parameters left from the impurity model :eq:`SIAM` are the one particle energy level of the bath
site :math:`\epsilon_c` and the hybridization strength :math:`V`.

Thus the hybridization function reduces to :math:`\Delta(\omega)=V^2/(\omega+ \mu -\epsilon_c)`

and the free :math:`(U=0)` impurity Green function is a two pole function

.. math::
   G^{(0)}_{imp} =& \frac{\omega - \epsilon_c + \mu}{(\omega + \mu - \epsilon_d)(\omega + \mu - \epsilon_c) - V^2} \\
    =& \frac{1}{2r} \left( \frac{r-\delta\epsilon}{\omega + \mu - \bar{\epsilon} +r}
    + \frac{r + \delta\epsilon}{\omega + \mu - \bar{\epsilon} - r} \right)

where :math:`\bar{\epsilon}=(\epsilon_d+\epsilon_c)/2`, :math:`\delta{\epsilon}
=(\epsilon_d-\epsilon_c)/2` and :math:`r=\sqrt{\delta \epsilon^2 + V^2}`. The
interacting Green function has four poles and the self-energy two poles.

Self-consistency
''''''''''''''''

In the Two site DMFT original self-consistency equations need to be reformulated
to be able to capture the desired physical behavior of the system in a such
simplified model. This mean to find two physically motivated conditions to fix
the bath parameters :math:`\epsilon_c` and :math:`V`.

In the limit of high frequencies the exact self-energy of the impurity problem
:eq:`SIAM` can be expanded in powers of :math:`1/\omega`:

.. math:: \Sigma(\omega) = Un_d + \frac{U^2 n_d (1 - n_d)}{\omega}
           + \mathcal{O}(1/\omega^2)
   :label: High_w_sigma_expan

where :math:`n_d` is the average occupancy of the impurity orbital:

.. math:: n_d = \braket{d^\dagger_\sigma d_\sigma} = - \frac{1}{\pi}
   \int_{-\infty}^0 \Im m G_{imp}(\omega+ i0^+) d\omega

Inserting the expansion :eq:`High_w_sigma_expan` into equation :eq:`Site_GF_H_Trans`
allows to find the high-frequency expansion of the on-site lattice Green function

.. math:: G(\omega) =&
    \frac{1}{\omega} + \frac{\epsilon_d - \mu + U n_d}{\omega^2} \\
    & + \frac{M_2^{(0)} + (\epsilon_d - \mu)^2 + 2(\epsilon_d -\mu)U n_d
              U^2 n_d}{\omega^3} + \mathcal{O}(1/\omega^4)
    :label: High_w_sigma_expan_High_G

where :math:`M_2^{(0)}=\\int  x^2 \\rho_0(x)dx` is the variance of the non-
interacting density of states :eq:`Bethe_dos`.
