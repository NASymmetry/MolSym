import numpy as np
import qcelemental as qcel
from dataclasses import dataclass
from copy import deepcopy
import sys
global_tol = 1e-8 # TODO It would be nice to get rid of this...

@dataclass
class Atom():
    """
    Dataclass for storing atom information

    :param Z: Atomic number of atom.
    :param mass: Mass of atom in amu as defined by QCElemental.
    :param xyz: Position vector of atom in Cartesian coordinates.
    :type Z: int 
    :type mass: float
    :type xyz: NumPy array of shape (3,)
    """
    Z:int
    mass:float
    xyz:np.array

@dataclass
class SEA():
    """
    SEA: symmetry equivalent atoms.
    SEAs are atoms that can be swapped with no distinguishable change in the molecule.

    :param label: Optionally defines rotor type of SEA set (e.g. Single Atom, Linear, Spherical, Regular Polygon, Oblate Symmetric Top, etc.)
    :param subset: Sublist of atom indices in molecule that constitute the SEA set
    :param axis: Optionally defines possible rotational symmetry vector
    :type label: str or None
    :type subset: NumPy array of integers
    :type axis: NumPy array of shape (3,) or None
    """
    label:str
    subset:np.array
    axis:np.array

class Molecule():
    """
    Class dealing with molecule relevant information.
    Typically initiated from a QCSchema object.
    """
    def __init__(self, atoms, coords, masses) -> None:
        self.tol = 1e-5
        self.atoms = np.asarray(atoms)
        try:
            self.natoms = len(self.atoms)
        except TypeError:
            self.natoms = 1
        self.coords = np.asarray(coords)
        self.masses = np.asarray(masses)

    @classmethod
    def from_schema(cls, schema):
        """
        Class method for constructing a Molecule from a QCSchema object

        :param schema: Schema dictionary to be converted to Molecule
        :type schema: dict
        :rtype: molsym.Molecule
        """
        atoms = schema["symbols"]
        natoms = len(atoms)
        coords = np.reshape(schema["geometry"], (natoms,3))
        # As of now, QCElemental seems to have issues assigning masses, so I do it
        masses = np.zeros(natoms)
        for (idx, symb) in enumerate(atoms):
            masses[idx] = qcel.periodictable.to_mass(symb)
        return cls(atoms, coords, masses)
    
    @classmethod
    def from_psi4_molecule(cls, mol):
        """
        Class method for constructing a Molecule from a QCSchema object

        :param schema: Schema dictionary to be converted to Molecule
        :type schema: dict
        :rtype: molsym.Molecule
        """
        if "psi4" not in sys.modules:
            raise ImportError("Psi4 is required to use this function")
        atoms = [mol.symbol(i) for i in range(mol.natom())]
        coords = mol.geometry().to_array()
        masses = [mol.mass(i) for i in range(mol.natom())]
        return cls(atoms, coords, masses)

    @classmethod
    def from_file(cls, fn, keep_angstrom=False):
        """
        Class method for constructing a Molecule from an *.xyz file

        :param fn: Filename
        :type fn: str
        :rtype: molsym.Molecule
        """
        with open(fn, "r") as lfn:
            strang = lfn.read()
        
        schema = qcel.models.Molecule.from_data(strang).dict()
        if keep_angstrom:
            schema["geometry"] *= qcel.constants.bohr2angstroms
        return cls.from_schema(schema)

    @classmethod
    def from_psi4_schema(cls, schema):
        """
        Class method for constructing a Molecule from a QCSchema object generated in Psi4.
        Schemas coming from Psi4 are different for some reason?

        :param schema: Schema dictionary to be converted to Molecule
        :type schema: dict
        :rtype: molsym.Molecule
        """
        atoms = schema["elem"] # was symbols
        natoms = len(atoms)
        coords = np.reshape(schema["geom"], (natoms,3)) # was geometry
        # As of now, QCElemental seems to have issues assigning masses, so I do it
        masses = np.zeros(natoms)
        for (idx, symb) in enumerate(atoms):
            masses[idx] = qcel.periodictable.to_mass(symb)
        return cls(atoms, coords, masses)

    def to_xyz_string(self, already_angstrom=False):
        # Will save xyz in Angstrom, undoing the previous
        # Ang->Bohr from Molecule.from_schema
        if already_angstrom:
            self.coords /= qcel.constants.bohr2angstroms
        qcmol = qcel.models.Molecule(
            **{"symbols": self.atoms, 
            "geometry": self.coords})
        return qcmol.to_string("xyz")

    def __repr__(self) -> str:
        rstr = "MolSym Molecule:\n"
        for i in range(self.natoms):
            rstr += f"   {self.atoms[i]:3s}   {self.coords[i,0]:12.8f}"
            rstr += f"   {self.coords[i,1]:12.8f}   {self.coords[i,2]:12.8f}\n"
        return rstr

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, i):
        return Molecule(self.atoms[i], self.coords[i,:], self.masses[i])

    def __len__(self):
        return self.natoms

    def __eq__(self, other):
        # Select higher tolerance
        if self.tol >= other.tol:
            eq_tol = self.tol
        else:
            eq_tol = other.tol
        if isinstance(other, Molecule):
            c1 = (other.atoms == self.atoms).all()
            c2 = (other.masses == self.masses).all()
            c3 = np.allclose(other.coords, self.coords, atol=eq_tol)
            return c1 and c2 and c3

    def find_com(self):
        """
        Get center of mass of molecule.

        :return: Center of mass
        :rtype: NumPy array of shape (3,)
        """
        com = np.zeros(3)
        for i in range(self.natoms):
            com += self.masses[i]*self.coords[i,:]
        return com / sum(self.masses)

    def is_at_com(self):
        """
        Checks if molecule is at center of mass already.

        :rtype: bool
        """
        if sum(abs(self.find_com())) < self.tol:
            return True
        else:
            return False

    def translate(self, r):
        """
        Translates Cartesian positions of all atoms in molecule in place by vector r.

        :param r: Translation vector
        :type r: NumPy array of shape (3,)
        """
        for i in range(self.natoms):
            self.coords[i,:] -= r
        
    def transform(self, M):
        """
        Transform coordinates of molecule by matrix M and return new molecule.

        :param M: Transformation matrix (e.g. rotation, reflection, etc.)
        :type M: NumPy array (3,3)
        :return: Molecule with transformed atom coordinates
        :rtype: molsym.Molecule
        """
        new_mol = deepcopy(self)
        new_mol.coords = np.dot(new_mol.coords,np.transpose(M))
        return new_mol

    def distance_matrix(self):
        """
        Calculates the interatomic distance matrix as all pairwise distances between atoms.

        :return: Interatomic distance matrix
        :rtype: NumPy array of shape (self.natoms,self.natoms)
        """
        dm = np.zeros((self.natoms,self.natoms))
        for i in range(self.natoms):
            for j in range(i,self.natoms):
                dm[i,j] = np.sqrt(sum((self.coords[i,:]-self.coords[j,:])**2))
                dm[j,i] = dm[i,j]
        return dm

    def find_SEAs(self):
        """
        Find sets of symmetry equivalent atoms.
        Permutations of the distance matrix reveal which atoms form symmetry equivalent sets.

        :return: List of symmetry equivalent atom sets
        :rtype: List[molsym.SEA]
        """
        dm = self.distance_matrix()
        out = []
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                a_idx = np.argsort(dm[i,:])
                b_idx = np.argsort(dm[j,:])
                z = dm[i,a_idx] - dm[j,b_idx]
                chk = True
                for k in z:
                    if abs(k) < self.tol:
                        continue
                    else:
                        chk = False
                if chk:
                    out.append((i,j))
        skip = []
        SEAs = []
        for i in range(self.natoms):
            if i in skip:
                continue
            else:
                collect = [i]
            
            for k in out:
                if i in k:
                    if i == k[0]:
                        collect.append(k[1])
                        skip.append(k[1])
                    else:
                        collect.append(k[0])
                        skip.append(k[0])
            SEAs.append(SEA("", collect, np.zeros(3)))
        return SEAs

