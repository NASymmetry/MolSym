import numpy as np
import re
import qcelemental as qcel
from molsym.molecule import Molecule
from molsym import find_point_group
from .point_group import PointGroup
from .general_irrep_mats import pg_to_symels
from .symtext_helper import get_atom_mapping, rotate_mol_to_symels, get_linear_atom_mapping, get_class_name
from .multiplication_table import build_mult_table, subgroup_by_name, subgroup_axes, multiply, inverse

class Symtext():
    """
    Fundamental object of MolSym, holds most of the symmetry information of the molecule.
    """
    def __init__(self, mol, rotate_to_std, reverse_rotate, pg, symels, atom_map, mult_table, irreps, irrep_mats) -> None:
        self.mol = mol
        self.rotate_to_std = rotate_to_std
        self.reverse_rotate = reverse_rotate
        self.pg = pg
        self.complex = False
        if self.pg.family == "C" and self.pg.n and self.pg.n > 2:
            if self.pg.subfamily is None or self.pg.subfamily == "h":
                self.complex = True
        elif self.pg.family == "S":
            self.complex = True
        elif self.pg.str in ["T", "Th"]:
            self.complex = True
        self.symels = symels
        self.atom_map = atom_map
        self.mult_table = mult_table
        if pg.is_linear:
            # Haar measure
            if pg.family == "C":
                self.order = 4*np.pi
            elif pg.family == "D":
                self.order = 8*np.pi
        else:
            self.order = len(symels)
        self.irreps = irreps
        self.irrep_mats = irrep_mats
        self.get_character_table()

    def __len__(self):
        return len(self.symels)

    def __repr__(self):
        return f"\n{self.mol}\n{self.character_table}\n{self.symels}\nAtom map:\n{self.atom_map}\nMultiplication Table\n{self.mult_table}"

    @classmethod
    def from_molecule(cls, mol):
        """
        Class method for creating a Symtext from a molsym.Molecule

        :type mol: molsym.Molecule
        :rtype: molsym.Symtext
        """
        mol.translate(mol.find_com())
        pg_str, (paxis, saxis) = find_point_group(mol)
        pg = PointGroup.from_string(pg_str)
        # Return transformation matrix so properties can be rotated to original configuration
        mol, reverse_rotate, rotate_to_std = rotate_mol_to_symels(mol, paxis, saxis)
        symels, irreps, irrep_mats = pg_to_symels(pg.str)
        if pg.is_linear:
            atom_map = get_linear_atom_mapping(mol, pg)
            return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, atom_map, None, irreps, irrep_mats)
        atom_map = get_atom_mapping(mol, symels)
        mult_table = build_mult_table(symels)
        return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, atom_map, mult_table, irreps, irrep_mats)

    @classmethod
    def from_file(cls, fn):
        """
        Class method for creating a Symtext from a *.xyz file.

        :type fn: str
        :rtype: molsym.Symtext
        """
        with open(fn, "r") as lfn:
            strang = lfn.read()
        schema = qcel.models.Molecule.from_data(strang).dict()
        mol = Molecule.from_schema(schema)
        return cls.from_molecule(mol)
    
    def get_character_table(self):
        """
        Get character table data. Characters are calculated from the irreducible representation matrices, classes are determined from the multiplication table.
        """
        # Sort classes
        self.classes = []
        self.symel_to_class_map = [0 for i in range(len(self))]
        self.class_orders = []
        done = []
        for sidx, symel in enumerate(self.symels):
            if sidx in done:
                continue
            else:
                cc = []
                for sidx2, symel2 in enumerate(self.symels):
                    cc.append(multiply(self.mult_table, sidx2, sidx, inverse(self.mult_table, sidx2)))
                reduced = list(set(cc))
                for r in reduced:
                    self.symel_to_class_map[r] = len(self.classes)
                reduced.reverse()
                self.classes.append(get_class_name([self.symels[i] for i in reduced]))
                self.class_orders.append(len(reduced))
                done += reduced

        if self.complex:
            self.character_table = np.zeros((len(self.irreps), len(self.classes)), dtype=np.complex128)
        else:
            self.character_table = np.zeros((len(self.irreps), len(self.classes)))
        for irrep_idx, irrep in enumerate(self.irreps):
            for class_idx, class_name in enumerate(self.classes):
                self.character_table[irrep_idx,class_idx] = np.trace(self.irrep_mats[irrep.symbol][self.symel_to_class_map.index(class_idx)])

    def direct_product(self, *args):
        """
        Return direct product of irrep indices (*args).

        :type *args: int
        :rtype: NumPy array of shape (n,)
        """
        out = self.character_table[args[0],:]
        for arg in args[1:]:
            out = np.multiply(out, self.character_table[arg,:])
        return out

    def direct_product2(self, *args):
        """
        Return direct product of irrep objects (*args).

        :type *args: molsym.Irrep
        :rtype: NumPy array of shape (n,)
        """
        new_args = [self.irreps.index(arg) for arg in args]
        return self.direct_product(*new_args)

    def contains_symmetric_irrep(self, dp_vector):
        """
        Return True if a direct product contains the totally symmetric irreducible representation.

        :type dp_vector: NumPy array of shape (n,)
        :rtype: bool
        """
        p = np.multiply(dp_vector, self.class_orders)
        return round(p.sum()/(self.order)) > 0


    def reduction_coefficients(self, rrep_characters):
        """
        Return reduction coefficients for the characters of a representation.

        :param rrep_characters: Characters for each class for a representation (reducible or irreducible)
        :type rrep_characters: NumPy array of shape (n,)
        :return: Integer array with number of occurences of each irrep. in rrep_characters.
        :rtype: NumPy array of shape (len(self.irreps),)
        """
        out = np.zeros(len(self.irreps), dtype=int)
        for irrep_idx, irrep in enumerate(self.irreps):
            p = np.multiply(rrep_characters, self.class_orders)
            p = np.multiply(p, self.character_table[irrep_idx,:])
            out[irrep_idx] = round(p.sum()/(self.order))
        return out

    @property
    def rotational_symmetry_number(self):
        """
        Returns the symmetry number for the rotational partition function.

        :rtype: int
        """
        if self.pg.family == "C":
            if self.pg.n == 0 or self.pg.n is None:
                return 1
            else:
                return self.pg.n
        elif self.pg.family == "D":
            if self.pg.n == 0:
                return 2
            else:
                return 2*self.pg.n
        elif self.pg.family == "S":
            return self.pg.n >> 1
        elif self.pg.family == "T":
            return 12
        elif self.pg.family == "O":
            return 24
        elif self.pg.family == "I":
            return 60

    def subgroup_symtext(self, subgroup_str):
        """
        Build a new Symtext for the specified subgroup from the parent Symtext.

        :param subgroup_str: Schoenflies symbol of desired subgroup.
        :type subgroup_str: str
        :rtype: molsym.Symtext
        """
        subgroup = PointGroup.from_string(subgroup_str)
        subgroup_symels, subgroup_irreps, subgroup_irrep_mats = pg_to_symels(subgroup.str)
        mult_table = build_mult_table(subgroup_symels)
        isomorphism = subgroup_by_name(self.symels, self.mult_table, subgroup.str)
        if isomorphism is None:
            raise Exception(f"No {subgroup.str} subgroup found for {self.pg} group")
        sgp = [self.symels[i[1]] for i in isomorphism]
        paxis, saxis = subgroup_axes(subgroup.str, sgp)
        new_mol, reverse_rotate, rotate_to_std = rotate_mol_to_symels(self.mol, paxis, saxis)
        new_mol.tol = self.mol.tol
        atom_map = get_atom_mapping(new_mol, subgroup_symels)
        return Symtext(new_mol, rotate_to_std, reverse_rotate, subgroup, subgroup_symels, atom_map, mult_table, subgroup_irreps, subgroup_irrep_mats)
    
    def largest_D2h_subgroup(self):
        """
        Build a new Symtext for the largest subgroup of the parent Symtext that is also a subgroup of the D2h point group.
        Some groups may have equivalent order subgroups, if you want a specific one, don't use this function

        :rtype: molsym.Symtext
        """
        D2h_subgroups = ["D2h", "D2", "C2v", "C2h", "Cs", "C2", "Ci", "C1"]
        for i in D2h_subgroups:
            try:
                return self.subgroup_symtext(i)
            except:
                pass
