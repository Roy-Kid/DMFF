from openmm.app import *
from openmm import *
from openmm.unit import *
from dmff.api import Hamiltonian
import jax.numpy as jnp
import jax
from jax import vmap, jit, value_and_grad
import optax
import dmff.settings as settings
from dmff.common.nblist import NeighborList
settings.DO_JIT = False

NA = 6.02*1E23

# load data
reportInterval = 10000
totalStep = 1E6
totalH2O = 996
n_report = totalStep / reportInterval
H2O_mass = 15.99943 + 1.007947*2
nH2Os = 996

T = 305
traj_310K = jnp.load(f'data/{T}K_traj.npy')
box_310K = jnp.load(f'data/{T}K_box.npy')
energy_310K = jnp.load(f'data/{T}K_energy.npy')
density_310K = jnp.load(f'data/{T}K_density.npy')

T = 300
traj_300K = jnp.load(f'data/{T}K_traj.npy')
box_300K = jnp.load(f'data/{T}K_box.npy')
energy_300K = jnp.load(f'data/{T}K_energy.npy')
density_300K = jnp.load(f'data/{T}K_density.npy')

from dmff.common.estimator import Reweighting

def target_func(x, box, params):
    """
    The physical quantity function $A=A(x, \tau)$ given by the user. In this example, we use the density as the target function.
    """
    x = box[0, 0] * 10*1E-8  # from nm to cm
    y = box[1, 1] * 10*1E-8
    z = box[2, 2] * 10*1E-8
    
    density = (H2O_mass * nH2Os / NA) / (x * y* z)
    return density

pdb = PDBFile('data/waterbox.pdb')
h = Hamiltonian('data/tip3p.xml')
ffparams = h.paramtree
potential = h.createPotential(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=4*angstrom, constraints=HBonds)
print(f'setup done')
generators = h.getGenerators()
energy_func = potential.getPotentialFunc()  # (pos, box, pairs, h.getGenerators()[0].paramtree)


reweight = Reweighting(style='npt')
# reweight.set_samples(atoms, box)
ensemble_params = {'T': 300., 'p': 1.}
reweight.set_ensemble_params(ensemble_params)  # only emsemble params, ff params from xml
reweight.set_ff_params(ffparams)
reweight.set_target_func(target_func)
reweight.set_energy_func(energy_func)

# warming up
nblist = NeighborList(8)
nblist.allocate(traj_300K[0], box_300K[0])

def fit(params:optax.Params, optimizer:optax.GradientTransformation)->optax.Params:
    
    opt_state = optimizer.init(params)
    
    def loss(params, label, positions, box, pairs):
        hat_A = reweight.estimate(params, positions, box, pairs)
        return (hat_A - label) ** 2
    
    def step(params, label, opt_state, position, box, pairs):
        loss_value, grads = value_and_grad(loss)(params, label, position, box, pairs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    target = 1.2
    for i, (position, box) in enumerate(zip(traj_300K, box_300K)):
        nblist.update(position, box)
        params, opt_state, loss_value = step(params, target, opt_state, position, box, nblist.pairs)
        print(f'step: {i}, loss: {loss_value}')
            
optimizer = optax.adam(learning_rate=1e-2)

params = {
    'ffparams': ffparams,
    'ensemble_params': ensemble_params
}

params = fit(params, optimizer)


# uncertainty = reweight.calc_uncertainty(density_310K)

# print(f'300K density: ', jnp.mean(density_300K))
# print(f'305K density: ', jnp.mean(density_310K))

# print('true value: ', jnp.mean(vmap(target_func, in_axes=(0, 0, None))(traj_310K, box_310K, None)))
# print('estimate value: ', hat_A)
# print('uncertainty', uncertainty)