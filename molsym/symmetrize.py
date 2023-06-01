import numpy as np
from .symtext.symtext import Symtext

def symmetrize(mol_in, asym_tol=0.05):
    mol_in.tol = asym_tol
    seas = mol_in.find_SEAs()
    asym_symtext = Symtext.from_molecule(mol_in)
    mol = asym_symtext.mol
    for sea in seas:
        atom_i = sea.subset[0]
        for g in range(1,asym_symtext.order):
            if atom_i == asym_symtext.atom_map[atom_i, g]:
                # Atom i invariant under g
                if asym_symtext.symels[g].symbol == "i":
                    # Atom i must be at origin
                    mol.coords[atom_i,:] = np.array([0.0,0.0,0.0])
                    break
                elif asym_symtext.symels[g].symbol[0] == "S":
                    # Atom i must be at origin
                    mol.coords[atom_i,:] = np.array([0.0,0.0,0.0])
                    break
                elif asym_symtext.symels[g].symbol[0] == "C":
                    # Project atom i onto rotation axis
                    l = np.dot(mol.coords[atom_i,:],asym_symtext.symels[g].vector)
                    mol.coords[atom_i,:] = l * asym_symtext.symels[g].vector
                elif asym_symtext.symels[g].symbol[:5] == "sigma":
                    # Project atom i onto plane
                    l = np.dot(mol.coords[atom_i,:],asym_symtext.symels[g].vector)
                    mol.coords[atom_i,:] -= l*asym_symtext.symels[g].vector
                else:
                    raise Exception("Wut")
        # Place other atoms in SEA based off of atom i
        for atom_j in sea.subset[1:]:
            for g in range(1,asym_symtext.order):
                if atom_j == asym_symtext.atom_map[atom_i, g]:
                    mol.coords[atom_j,:] = np.dot(asym_symtext.symels[g].rrep, mol.coords[atom_i,:])
                    break

    mol.tol = 1e-8
    return mol
