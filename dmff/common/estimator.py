# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-10
# version: 0.0.1

from copy import deepcopy
from typing import Callable, Dict, Literal
from jax import vmap
import jax.numpy as jnp
import numpy as np
from dmff.common.nblist import NeighborList

kb = 1.38064852e-23

class Reweighting:

    def __init__(self, positions, box):

        self.target_func:Callable = None
        self.set_samples(positions, box)
        self.ff_params:Dict = None
        self.ensemble_params:Dict = None

    def set_target_func(self, target_func:Callable):
        r"""
        Physical quantity function $A=A(x, \tau)$ given by the user. 

        Parameters
        ----------
        target_func : Callable
            The physical quantity function of ensemble average depends on ensemble and sampling parameters
        """
        self.target_func = target_func

    def set_samples(self, samples:jnp.ndarray, box:jnp.ndarray):
        r"""
        set a series of trajectories ${x_n}, n=1...N$ of the ensemble to this estimator. The physical quantity function $A=A(x, \tau)$ are rely on the ensemble. 

        Parameters
        ----------
        samples : jnp.ndarray
            The shape of samples should be `(N, N_a, 3)`, N is the length of trajectories, N_a is the number of particle members, and 3 is the dimension of the state.
        """
        
        assert samples.ndim == 4

        self.samples = samples
        self.box = box
        self.n_samples = len(self.positions)
        assert self.n_samples == len(self.box)

    def set_ff_params(self, params:Dict):
        r"""
        set the sampling parameters $\Tau_0={\theta_0, \alpha_0}$. 

        Parameters
        ----------
        params : Dict
            a dict of sampling parameters.
        """

        self.ff_params = params

    def set_ensemble_params(self, params:Dict):
        self.ensemble_params = params
        self.T = params['T']
        self.beta = kb * self.T

    def set_energy_func(self, energy_func:Callable)->Callable:
        r"""
        set the energy function $E=E(x, \tau)$ given by the user. 

        Parameters
        ----------
        energy_func : Callable
            The energy function of ensemble average depends on ensemble and sampling parameters
        """
        self.energy_func = energy_func

    def estimate(self, ensemble_style:Literal['npt', 'nve', 'nvt'], ffparams:Dict, ensemble_params:Dict):
        r"""
        return the rewighting estimator of the physical quantity function $\hat{A_0}(\Tau_1)$

        Parameters
        ----------
        params : Dict
            $\tau_1$
        """
        theta0 = self.ff_params
        theta1 = deepcopy(self.ff_params)
        theta1.update(ffparams)
        alpha0 = self.ensemble_params
        alpha1 = deepcopy(self.ensemble_params)
        alpha1.update(ensemble_params)

        calc_energy = self.energy_func

        T0 = alpha0['T']
        T1 = alpha1['T']
        beta0 = kb * T0
        beta0 = 1/beta0
        beta1 = kb * T1
        beta1 = 1/beta1
        P0 = alpha0['pressure']
        P1 = alpha1['pressure']

        # warm up
        nblist = NeighborList(2.5)
        nblist.allocate(self.positions[0], self.box[0])

        def calc_nblist(positions, box):
            nblist.update(positions, box)
            return nblist.pairs

        pairs = vmap(calc_nblist, in_axes=(0, 0))(self.positions, self.box)
        pairs = np.array(pairs)

        def _hat_A(position, box, pairs):

            # calculate energy
            V = box[0, 0] * box[1, 1] * box[2, 2]
            u0 = calc_energy(position, box, pairs, theta0) + P0*V
            u1 = calc_energy(position, box, pairs, theta1) + P1*V
            d_blz = -( (beta0 * u0) - (beta1 * u1) )
            offset = jnp.max(d_blz)
            d_blz = jnp.exp(d_blz - offset)

            fenzi = self.target_func(position, box, theta1) * d_blz
            fenmu = d_blz
            return fenzi, fenmu

        fenzis, fenmus = vmap(_hat_A, in_axes=(0, 0, 0))(self.positions, self.box, pairs)
        self._fenzis = fenzis
        self._fenmus = fenmus

        hat_A = jnp.sum(fenzis) / jnp.sum(fenmus)
        self._hat_A = hat_A
        return hat_A

    def __call__(self, *args, **kwargs):
        return self.estimate(*args, **kwargs)

    def calc_uncertainty(self):

        prefactor = 1 / (self._hat_A * self.n_samples)**2
        uncertainty = jnp.sum(vmap(lambda An: (self._hat_A - An)**2, in_axes=(0))(self._fenzis / self._fenmus)) * prefactor

        return uncertainty

    uncertainty = property(calc_uncertainty)

    def resample(self):
        pass
    