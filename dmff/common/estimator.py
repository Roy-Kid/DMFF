# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-10
# version: 0.0.1

from copy import deepcopy
from typing import Callable, Dict, Literal
from jax import vmap, tree_util
import jax.numpy as jnp
import numpy as np
from dmff.common.nblist import NeighborList

kb = 8.314  # J/K/mol

class Reweighting:

    def __init__(self, style):

        self.target_func:Callable = None
        self.ff_params:Dict = None
        self.ensemble_params:Dict = None
        self.style = style

    def set_target_func(self, target_func:Callable):
        r"""
        Physical quantity function $A=A(x, \tau)$ given by the user. 

        Parameters
        ----------
        target_func : Callable
            The physical quantity function of ensemble average depends on ensemble and sampling parameters
        """
        self.target_func = target_func

    # def set_samples(self, traj:jnp.ndarray, box:jnp.ndarray):
    #     r"""
    #     set a series of trajectories ${x_n}, n=1...N$ of the ensemble to this estimator. The physical quantity function $A=A(x, \tau)$ are rely on the ensemble. 

    #     Parameters
    #     ----------
    #     traj : jnp.ndarray
    #         The shape of traj should be `(N, N_a, 3)`, N is the length of trajectories, N_a is the number of particle members, and 3 is the dimension of the state.
    #     """

    #     assert len(box) == len(traj)
    #     self.traj = traj
    #     self.box = box
    #     self.n_traj = len(traj)
        
    #     # warm up
    #     self.nblist = NeighborList(8)
    #     self.nblist.allocate(self.traj[0], self.box[0])

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

    def estimate(self, params, positions, box, pairs):
        r"""
        return the rewighting estimator of the physical quantity function $\hat{A_0}(\Tau_1)$

        Parameters
        ----------
        params : Dict
            $\tau_1$
        """
        if positions.ndim == 2:
            positions = positions.reshape(-1, *positions.shape)  # (N, natoms, 3)
        if box.ndim == 2:
            box = box.reshape(-1, *box.shape)  # (N, 3, 3)
                
        theta0 = self.ff_params
        theta1 = params['ffparams']

        alpha0 = self.ensemble_params
        alpha1 = params['ensemble_params']
            
        calc_energy = self.energy_func
        
        # extract ensemble parameters
        T0 = alpha0['T']
        T1 = alpha1['T']
        beta0 = 1/ (kb * T0)
        beta1 = 1/ (kb * T1)
        P0 = alpha0['p']
        P1 = alpha1['p']
        
        ensemble_style = self.style

        if ensemble_style == 'npt':


            def _estimate(position, box):

                # calculate energy
                V = box[0, 0] * box[1, 1] * box[2, 2]
                u0 = calc_energy(position, box, pairs, theta0) + P0*V
                u1 = calc_energy(position, box, pairs, theta1) + P1*V
                d_blz = -( (beta0 * u0) - (beta1 * u1) )
                A = self.target_func(position, box, theta1)
                _exp = jnp.exp(d_blz)
                A_exp = A * _exp

                return A_exp, _exp
            
        elif ensemble_style == 'nvt':
            
            def _estimate(position, box):

                # calculate energy
                u0 = calc_energy(position, box, pairs, theta0)
                u1 = calc_energy(position, box, pairs, theta1)
                
                
                d_blz = -( (beta0 * u0) - (beta1 * u1) )
                A = self.target_func(position, box, theta1)

                _exp = jnp.exp(d_blz)
                A_exp = A * _exp

                return A_exp, _exp     
            
        elif ensemble_style == 'muvt':
            pass
            
        else:
            raise KeyError

        len_pos = len(positions)
        len_box = len(box)
        assert len_box == len_pos, f'len of positions {len_pos} != len of box {len_box}'


        # this method may causes OOM on GPU
        A_exps, _exps = vmap(_estimate, in_axes=(0, 0))(positions, box)

        # if so, use follow snippet instead
        # A_exps = jnp.zeros(len_pos)
        # _exps = jnp.zeros(len_pos)
        
        # for i in range(len_pos):
        #     A_exp, fenmu = _estimate(positions[i], box[i])
        #     A_exps = A_exps.at[i].set(A_exp)
        #     _exps = _exps.at[i].set(fenmu)
        
        hat_A = jnp.sum(A_exps) / jnp.sum(_exps)
        return hat_A

    def __call__(self, *args, **kwargs):
        return self.estimate(*args, **kwargs)

    def calc_uncertainty(self, true_value):

        prefactor = 1 / (self._hat_A * self.n_traj)**2

        def _uncertainty(An):
            # An is true value
            # _hat_A is estimate value
            return (self._hat_A - An)**2
        
        uncertainty = jnp.sum(vmap(_uncertainty, in_axes=(0))(true_value)) * prefactor

        return uncertainty

    def resample(self):
        pass
    