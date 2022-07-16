# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-10
# version: 0.0.1

from copy import deepcopy
from typing import Callable, Dict
from jax import vmap
import jax.numpy as jnp
import numpy as np
import freud
from dmff.common.nblist import NeighborList

kb = 1.38064852e-23

class Reweighting:

    def __init__(self, positions, box):

        self.target_func:Callable = None
        self.positions:jnp.ndarray = positions
        self.box:jnp.ndarray = box
        self.params:Dict = None

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

    def estimate(self, ffparams:Dict, ensemble_params:Dict):
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

        # warm up
        nblist = NeighborList(self.box[0], 2.5)
        nblist.allocate(self.positions[0])

        def calc_nblist(positions, box):
            nblist.update(positions, box)
            return nblist.pairs

        pairs = vmap(calc_nblist, in_axes=(0, 0))(self.positions, self.box)
        pairs = np.array(pairs)

        def _hat_A(position, box, pairs):

            # calculate energy
            u0 = calc_energy(position, box, pairs, theta0)
            u1 = calc_energy(position, box, pairs, theta1)
            d_blz = jnp.exp(-self.beta * (u1 - u0))

            fenzi = self.target_func(position, box, theta1) * d_blz
            fenmu = d_blz
            return fenzi, fenmu

        fenzis, fenmus = vmap(_hat_A, in_axes=(0, 0, 0))(self.positions, self.box, pairs)

        return jnp.sum(fenzis) / jnp.sum(fenmus)
                

    def __call__(self, *args, **kwargs):
        return self.estimate(*args, **kwargs)

    def calc_uncertainty(self):

        pass

    def resample(self):
        pass
    