import psi4
import numpy as np
from input import Settings
import molsym
from molsym import Molecule
from molsym.salcs.RSH_SALCs import Project

def qc(molecule):
    qc_obj = {
        "symbols": [molecule.symbol(x) for x in range(0, molecule.natom())] ,
        "geometry": molecule.geometry(),
    }
    return qc_obj

molecule = psi4.geometry(Settings['molecule'])
molecule.update_geometry()
ndocc = Settings['nalpha'] #Settings['nbeta']
scf_max_iter = Settings['scf_max_iter']

schema = qc(molecule)

mol = Molecule.from_schema(schema)
mol = molsym.symmetrize(mol)
symtext = molsym.Symtext.from_molecule(mol)
mole = symtext.mol
molecule.set_geometry(psi4.core.Matrix.from_array(symtext.mol.coords))
print("updated psi4 mol")
print(symtext.pg)

basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'], puream=True)
Enuc = molecule.nuclear_repulsion_energy()
ints = psi4.core.MintsHelper(basis)

#symtext = molsym.Symtext.from_molecule(mol)
#bset = [[0,0,1],[0],[0],[0]]
bset = molsym.salcs.RSH_SALCs.get_basis(molecule,basis)[0]
coords = molsym.salcs.SphericalHarmonics(symtext, bset)
my_salcs = molsym.salcs.ProjectionOp(symtext, coords)
#print(my_salcs.basis_transformation_matrix)

#guess = 'core'
#guess = 'gwh'
salcs, irreplength, reorder_psi4 = molsym.salcs.RSH_SALCs.Project(mole, molecule, basis, symtext)
print(irreplength)
np.set_printoptions(precision=9, threshold=np.inf)
for salc in salcs:
    print(salc.lcao)