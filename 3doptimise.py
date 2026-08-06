import ase 
from ase.io import read, write
from ase.optimize import LBFGS, FIRE
from upet.calculator import UPETCalculator


atoms = read("lutathera-opt.xyz")
calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cpu")
atoms.calc = calculator
dyn = LBFGS(atoms, maxstep=0.05, trajectory='history.traj')
dyn.run(fmax=0.01, steps=300)
write("lutathera-opt.xyz", atoms)
