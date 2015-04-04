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
   :label: ImpGF_free

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

where :math:`n_d\equiv n_{d\sigma}` is the spin specific average occupancy of the impurity orbital:

.. math:: n_d = \braket{d^\dagger_\sigma d_\sigma} = - \frac{1}{\pi}
   \int_{-\infty}^0 \Im m G_{imp}(\omega+ i0^+) d\omega

Inserting the expansion :eq:`High_w_sigma_expan` into equation :eq:`Site_GF_H_Trans`
allows to find the high-frequency expansion of the on-site lattice Green function

.. math:: G(\omega) =&
    \frac{1}{\omega} + \frac{\epsilon_d - \mu + U n_d}{\omega^2} \\
    & + \frac{M_2^{(0)} + (\epsilon_d - \mu)^2 + 2(\epsilon_d -\mu)U n_d
             + U^2 n_d}{\omega^3} + \mathcal{O}(1/\omega^4)
    :label: High_w_sigma_expan_High_G

where :math:`M_2^{(0)}=\int  x^2 \rho_0(x)dx` is the variance of the non-
interacting density of states :eq:`Bethe_dos`. This expansion has to relate the
fillings of the impurity :math:`n_{imp} \equiv 2 n_d` model with the lattice model as such it is required that the fillings
in both models match.

.. math:: n_{lattice} = n_{imp}
   :label: occupancy_match

where the band filling is calculated via

.. math:: n_{lattice} = - \frac{2}{\pi} \int_{-\infty}^0 \Im m G(\omega+ i0^+) d\omega
   :label: lattice_ocupation

Equation :eq:`occupancy_match` can be seen as an integral for of the original
self-consistency condition :eq:`DMFT_selfconsistency` and the paramagnetic
solution is enforced as the spin species are dealt equivalent.

The low-frequency limit of the self-energy can be expanded in powers of :math:`\omega`

.. math:: \Sigma(\omega) = a + b\omega +\mathcal{O}(\omega^2)
   :label: Low_w_sigma_expan

The definition :math:`z=1/(1-b)` of the quasiparticle weight for the metal behavior
of the system is convenient :math:`z=1/(1-d\Sigma(0)/d\omega)`. Inserting the
expansion :eq:`Low_w_sigma_expan` with the definition of the quasiparticle
weigh one obtains the coherent part of the on-site Green function:

.. math:: G^{coh}(\omega) = z \int_{-\infty}^{\infty} \frac{\rho_0(x)dx}{\omega -z(x-\mu+a)}
   :label: G_coh

On the other hand the coherent part of the impurity Green function is

.. math:: G^{coh}_{imp}(\omega) = \frac{z}{\omega - z(\epsilon_d - \mu + a + \Delta(\omega))}

If one where to compare the low frequency expansion of these coherent Green functions
one obtains

.. math::
    G^{coh}(\omega)= & \int_{-\infty}^{\infty} \frac{\rho_0(x)dx}{x - \mu + a}
    - \frac{\omega}{z}\int_{-\infty}^{\infty} \frac{\rho_0(x)dx}{(x - \mu + a)^2}
    + \frac{\omega^2}{z^2}\int_{-\infty}^{\infty} \frac{\rho_0(x)dx}{(x - \mu + a)^3} +\mathcal{O}(\omega^3) \\
    G^{coh}_{imp}(\omega)= & \frac{\mu -\epsilon_c}{k-V^2}  + \left(\frac{\mu -\epsilon_c}{k-V^2}
    - \frac{(\mu - \epsilon_c)(k' + k -  V^2)}{(k-V^2)^2}\right) \omega +\mathcal{O}(\omega^2)

where :math:`k'=2\mu - \epsilon_c - \epsilon_d + b` and
:math:`k=\mu^2 -2\mu(\epsilon_c + \epsilon_d) + \epsilon_c\epsilon_d -a`. Since
it becomes to complicated to find a link in these low frequency expansions, one
performs the high-frequency expansion of the coherent Green functions to obtains

.. math::
    G^{coh}(\omega) =& \frac{z}{\omega} + \frac{z^2(\epsilon_d - \mu +a)}{\omega^2}
    +\frac{z^3(M_2^{(0)} + (\epsilon_d - \mu +a)^2)}{\omega^3} \\
    G^{coh}_{imp}(\omega)= & \frac{z}{\omega} + \frac{z^2(\epsilon_d - \mu +a)}{\omega^2}
    +\frac{z^2V^2 + z^3(\epsilon_d - \mu +a)^2}{\omega^3} \\

leading to the second self-consistency condition

.. math:: V^2 = z M_2^{(0)}
   :label: hybridization_match

Algorithm implementation
''''''''''''''''''''''''

Using the two self-consistency conditions :eq:`occupancy_match` and :eq:`hybridization_match`
the bath parameters can be fixed and calculated self-consistently. One starts with
the model parameters :math:`\epsilon_d=0, t, U, \mu, \rho_0(x)` and takes a guess
for :math:`\epsilon_c, V^2`. That defines the two-site impurity model and can be
solved to find de average occupancy of the impurity :math:`n_{imp}=\braket{n_\uparrow}+\braket{n_\downarrow}`
and using the Lehmann representation one finds :math:`G_{imp}`, through the Dyson
equation one can extract the self-energy.

The self-energy yields the quasiparticle weight and through :eq:`hybridization_match`
a new value for the hybridization strength :math:`V`. The self-energy is used
again in :eq:`Site_GF_H_Trans` to obtain the lattice Green function, which via
:eq:`lattice_ocupation` yields the filling of the lattice sites and has to be
compared to the impurity occupancy. Then a new value for :math:`\epsilon_c` is
chosen to reduce the difference in occupancies between lattice and impurity models.
This cycle is performed until the self-consistency conditions are full-filled.

It is inconvenient to calculate the lattice Green function on each iteration to
calculate later the lattice occupancy with :eq:`lattice_ocupation`, as the numerical
pole broadening :math:`i0^+` introduces a lot of numerical variation. Instead, given
that the self-energy is a real two poled function, and on the Bethe lattice is
purely local and momentum independent, the lattice filling can be directly calculated
by

.. math:: n= 2 \int_{-\infty}^0 d\omega \rho_0(\omega + \mu - \Sigma(\omega))

where :math:`\rho(\omega)=\rho_0(\omega + \mu - \Sigma(\omega))` becomes the
interacting density of states. This becomes much more favorable as only this real
integral has to be calculated instead of the much more expensive hilbert transform
of :eq:`Site_GF_H_Trans` and one does not need to include the line broadening at
all.


Results
-------

The Mott transition at half-filling
'''''''''''''''''''''''''''''''''''

For the symmetric case of half-filling particle-hole symmetry requires
:math:`\epsilon_c=\mu=U/2` to ensure the first self-consistency equation :eq:`occupancy_match`.
One then follows to calculate the state of the system as the local Coulomb interaction
is raised and one shows in the next figure the change of the quasiparticle
weight at different temperatures.

.. plot:: ../examples/twosite/plot_halffill_z.py plot_z_real

Following the system state as the metallic behavior is lost. Higher temperatures
are less capable of sustaining the metal solution and the system just drops into
the insulating state.

The evolution of the spectral function is demonstrated in the next figures,
accompanied by the plot of the self energy and the impurity Green function. It
can be clearly seen that the self-energy has in general only 2 poles and the
local Green function is in general a 4 poled function.

.. plot:: ../examples/twosite/plot_halffill.py

The structure of the spectral function is not as rich as a full DMFT calculation
would provide. Nevertheless the transfer of spectral weight for low to high
energy is clearly demonstrated with the formation of the upper an lower Hubbard
bands as the remaining quasiparticle peak. There is the clear evidence of the
3 peak structure. The two site DMFT reduces the lattice Green function at very
low temperatures to a function of 4 poles, 2 poles coming from the self-energy
and the other 2 from the free impurity Green function :eq:`ImpGF_free`.

Away of Half-filling
''''''''''''''''''''

The self-consistency equations become more complicated to fulfill, as now one
has to search for the parameters :math:`\epsilon_c,V` at the same time for
a given target population and for a given local interaction. In the next figure
it is shown the quasiparticle weight decay as a function of increasing filling
for different values of the local interaction. In the empty band limit
:math:`Z\rightarrow 1` for an arbitrary :math:`U`. As
long as the system is away from half-filling it has a metallic behavior.

.. plot:: ../examples/twosite/plot_dop_z.py plot_z

The next plot show the expected value of the double occupation in the impurity.
At low lattice filling, attibuted to the low electron population, double occupancy
is low. When the system is subject to a local Coulomb interaction :math:`U<U_c`
the double occupancy increases as the system approaches the half-filling regime
and remains finite. When :math:`U<U_c` there is no double occupancy any more
at half-filling.

.. plot:: ../examples/twosite/plot_dop_d.py plot_d

The next figure shows
this parameters as a function of filling for the case :math:`U/t=4`. In this
case :math:`U<U_c` the hybridization strength is finite for :math:`n=1` and the
system becomes less correlated for decreasing filling. Consequently :math:`V` has
to increase until :math:`V\rightarrow M_2^{(0)}` for (:math:`n=0`).

.. plot:: ../examples/twosite/plot_dop.py plot_doping_param

As for :math:`\epsilon_c=U/2` at half-filling and it decreases with decreasing
filling until it diverges on appraching the empty band limit where
:math:`\epsilon_c \rightarrow -\infty`, as is necessary to ensure a vanishing
occupancy of the impurity orbital for finite :math:`V`. For the usable chemical
potential to dope the system it is only spanned in the range of bandwidth of the
non interacting density of states.

The structure of the spectral function change under doping is presented in the
next figure. The strong simplification of the two site DMFT approximation allows
only the visualization of the 3 "peaks" which always present a disconnected
structure. The quasiparticle peak never joins the lower Hubbard band at low
fillings.

.. plot:: ../examples/twosite/plot_dop_A.py

When analyzing the double occupancy of the impurity site



In Matsubara frequencies
''''''''''''''''''''''''

The twosite DMFT can also be treated in the imaginary axis using the Matsubara
frequencies. Since one is only treating the analytic continuation into the
imaginary axis, all equations are kept the same one only needs to change
:math:`\omega \rightarrow i\omega_n`, where :math:`\omega_n = \pi(2n+1)/\beta`.
The most substantial change comes from the
definition of the quasiparticle weight were it becomes

.. math:: z = [1 - \Im m \Sigma(\omega_0)/\omega_0) ]^{-1}

Until now only the treatment at half-filling is handled, because in this situation
the particle-hole symmetry requires :math:`\epsilon_c=\mu=U/2` and :eq:`occupancy_match`
is automatically fulfilled. The next figure shows the drop in the quasiparticle
weight as the local interaction is raised. It is very similar to the one presented
for the real frequencies. First an most noticeable if that a :math:`\beta = 6`
the quasiparticle weight doesn't drop completely to zero at :math:`U/D=1.6`, but
that the metallic solution is sustained up to :math:`U/D=2`. In general the metallic
solutions are sustained for larger interaction values(although not directly noticeable)
that the calculations performed in the real frequency axis.

The search for the
coexistence region of the metallic and insulating solutions is not possible
in the 2 site DMFT approximation as one only uses equation :eq:`hybridization_match`
to update the bath hybridization and starting from the insulating solution with
:math:`z=0` does not allow the system to escape the insulating case.

.. plot:: ../examples/twosite/plot_halffill_z.py plot_z

The behavior of the Green function and the self-energy is much more abstract than
in the real case. Here there are no poles to be seen. The next figure shows the
imaginary parts of the Green function and self-energy at different local interaction
strength. In the case of the self-energy the behavior is the same know from my
previous training with IPT. For the Green function there was the behavior it should
go to the bandwidth value at at zero frequency, this know condition is not met
neither was it enforced. In the case of the 2 site DMFT the Green Function always
"goes" through zero although there is no zero frequency in the fermionic case.

.. plot:: ../examples/twosite/plot_halffill_matsubara.py
