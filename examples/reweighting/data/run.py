from openmm.app import *
from openmm import *
from openmm.unit import *
import openmm.app.element as elem
from sys import stdout, argv
import numpy as np
from mdtraj.reporters import NetCDFReporter

# setup parameters
totalSteps = 1E6
reportInterval = 10000
_T = 310
T = _T*kelvin

pdb = PDBFile('waterbox.pdb')

forcefield = ForceField('tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinIntegrator(T, 1/picosecond, 2*femtoseconds)
system.addForce(MonteCarloBarostat(1*atmosphere, T, 25))

# Make sure all the forces we expect are present
for force in range(system.getNumForces()):
    print(system.getForce(force))

simulation = Simulation(pdb.topology, system, integrator)
context = simulation.context
if pdb.topology.getPeriodicBoxVectors():
    context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Initialize
context.setPositions(pdb.positions)


# Dump trajectory info every 10ps
#simulation.reporters.append(DCDReporter('output.dcd', 5000))
# Dump simulation info every 1ps
simulation.reporters.append(NetCDFReporter(f'{_T}K.nc', reportInterval))
simulation.reporters.append(StateDataReporter(stdout, reportInterval, step=True, totalEnergy=True, temperature=True, density=True, progress=True, volume=True, totalSteps=totalSteps, separator='\t'))
simulation.step(totalSteps)