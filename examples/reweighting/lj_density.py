
import mdtraj as md
import numpy as np
from dmff.api import Hamiltonian
import openmm.app as app
import freud
import jax.numpy as jnp
from jax import vmap
import dmff.settings as settings
settings.DO_JIT = False

t = md.load_netcdf('data/output.nc', top='data/output.pdb')

atoms = jnp.array(t.xyz) # (n_frames, n_atoms, 3)
box = jnp.array(t.unitcell_vectors) # (n_frames, 3)

from dmff.common.estimator import Reweighting

def target_func(x, box, params):
    """
    The physical quantity function $A=A(x, \tau)$ given by the user. In this example, we use the density as the target function.
    """
    # particle mass is 18 from xml
    # but I don't know how to get it from our fftree
    mass = 18
    density = len(x) * mass / box[0,0] / box[1,1] / box[2,2]
    return density

pdb = app.PDBFile('data/output.pdb')
h = Hamiltonian('data/lj.xml')
potential = h.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)
generators = h.getGenerators()
ljgenerators = generators[0]
ffparams = ljgenerators.paramtree
ljE = potential.getPotentialFunc()  # (pos, box, pairs, h.getGenerators()[0].paramtree)

reweight = Reweighting(atoms, box)
# reweight.set_samples(atoms, box)
reweight.set_ensemble_params({'T': 300, 'pressure': 1})  # only emsemble params, ff params from xml
reweight.set_ff_params(ffparams)
reweight.set_target_func(target_func)
reweight.set_energy_func(ljE)
hat_A = reweight.estimate('npt', {}, {'T': 310})  # \tau_1
print(jnp.mean(vmap(target_func, in_axes=(0, 0, None))(atoms, box, None)))
print(hat_A)
uncertainty = reweight.uncertainty
print(uncertainty)