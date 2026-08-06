from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

mol = Chem.MolFromXYZFile("Lutathera.xyz")
rdDetermineBonds.DetermineBonds(mol, charge=0)  # guesses bonds + bond orders

# now you can get 2D coords
from rdkit.Chem import AllChem
AllChem.Compute2DCoords(mol)

Chem.MolToMolFile(mol, "molecule_2d.mol")
