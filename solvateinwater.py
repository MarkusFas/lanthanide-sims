import ase 
from ase.io import read, write
import numpy as np
import sys
import subprocess
from atomic_symbols import symbol_to_id
from argparse import ArgumentParser

def generate_packmol_input(box_lengths, num_water, mol_file="molecule.xyz", water_file="water.xyz", output_file="output.xyz", packmol_file="packmol.inp"):
    """
    Generates a Packmol input file to solvate a molecule with water.

    Parameters:
    box_size (float): Length of the cubic box in Angstroms.
    num_water (int): Number of water molecules to insert.
    mol_file (str): XYZ file of the small molecule.
    water_file (str): XYZ file of a single water molecule.
    output_file (str): Name of the Packmol input file.
    """

    packmol_template = f"""tolerance 2.0
filetype xyz
output {output_file}
pbc {box_lengths[0]} {box_lengths[1]} {box_lengths[2]}
structure {mol_file}
  number 1
  fixed {box_lengths[0]/2} {box_lengths[1]/2} {box_lengths[2]/2} 0. 0. 0. 
  center
end structure

structure {water_file}
  number {num_water}
  inside box 0. 0. 0. {box_lengths[0]} {box_lengths[1]} {box_lengths[2]}
end structure
"""

    with open(packmol_file, "w") as f:
        f.write(packmol_template)

    print(f"Packmol input file '{packmol_file}' generated successfully.")



def center_molecule(mol_file, box_lengths):
    mol = read(mol_file)
    # Center the molecule in the middle of the box
    mol.center(vacuum=0.0)
    # Shift molecule to the exact center of the box
    print(box_lengths / 2)
    print(mol.get_center_of_mass())
    print(box_lengths / 2 - mol.get_center_of_mass())
    mol.translate(box_lengths / 2 - mol.get_center_of_mass())
    write(mol_file, mol, format='xyz')

def get_num(vol):
    mass = rho*vol
    num = (mass/molmass) * 6.022E23
    num -= 1 # correcting for the volume taken up by the dota molecules
    return round(num)

if __name__ == "__main__":

    parser = ArgumentParser(description="Create a water box around a molecule using Packmol")
    parser.add_argument("solute_file", help="XYZ file of the small molecule or atom")
    parser.add_argument("water_file", help="XYZ file of a single water molecule")
    parser.add_argument("output_file", help="Output file for the solvated system")
    parser.add_argument("--box_length", type=float, default=20.0, help="Length of the cubic box in Angstroms (default: 30.0)")
    args = parser.parse_args()

    packmol_output_file = args.output_file
    packmol_file = "packmol.inp"
    rho = 0.997 #g/cm^3
    molmass = 18.015 #g/mol
    avo = 6.022E23
    box_lengths = np.array([args.box_length, args.box_length, args.box_length])  # Box size in Angstroms
    volume = box_lengths[0] * box_lengths[1] * box_lengths[2] * 1E-24 # cm^3 (box size given in Angstroms)
    #sphere_volume = 4/3 * np.pi * (9**3) * 1E-24 # radius of 10 Angstroms
    
    # generate # of water molecules based on the volume of the box minus the volume of the sphere around the molecule
    solute = read(args.solute_file)
    solute_mass = np.sum(solute.get_masses())
    N = int((avo *rho*volume - solute_mass)/molmass)
    
    center_molecule(args.solute_file, box_lengths)
    print(box_lengths, N, args.solute_file, args.water_file, packmol_output_file, packmol_file)
    generate_packmol_input(box_lengths, N, args.solute_file, args.water_file, packmol_output_file, packmol_file)
    with open(packmol_file, "r") as f:
        run = subprocess.run(["packmol"], stdin=f, capture_output=True, text=True)
        print(run.stdout)
    # save as lammps.data
    structures = read(packmol_output_file)
    structures.set_cell(box_lengths)
    structures.set_pbc(True)
    
    write(args.output_file, structures, format='extxyz')
    write("lammps_indexing.data", structures, format='lammps-data')
    #write("lammps_indexing.xyz", structures, format='extxyz')
    print(f"Packmol run completed. Output written to {args.output_file}.")
    
