import qcelemental as qcel
from molsym.molecule import Molecule
from molsym import find_point_group
from .main import *
from .multiplication_table import build_mult_table, subgroup_by_name, subgroup_axes

class Symtext():
    def __init__(self, mol, rotate_to_std, reverse_rotate, pg, symels, chartable, class_map, atom_map, mult_table) -> None:
        self.mol = mol
        self.rotate_to_std = rotate_to_std
        self.reverse_rotate = reverse_rotate
        self.pg = PointGroup.from_string(pg)
        self.complex = False
        if self.pg.family == "C" and self.pg.n and self.pg.n > 2:
            if self.pg.subfamily is None or self.pg.subfamily == "h":
                self.complex = True
        elif self.pg.family == "S":
            self.complex = True
        elif self.pg.str in ["T", "Th"]:
            self.complex = True
        self.symels = symels
        self.chartable = chartable
        self.class_map = class_map
        self.atom_map = atom_map
        self.mult_table = mult_table
        self.order = len(symels)

    def __len__(self):
        return len(self.symels)

    def __repr__(self):
        return f"\n{self.mol}\n{self.chartable}\n{self.symels}\nClass map:\n{self.class_map}\nAtom map:\n{self.atom_map}\nMultiplication Table\n{self.mult_table}"

    @classmethod
    def from_molecule(cls, mol):
        mol.translate(mol.find_com())
        pg, (paxis, saxis) = find_point_group(mol)
        symels = pg_to_symels(pg)
        # Return transformation matrix so properties can be rotated to original configuration
        mol, reverse_rotate, rotate_to_std = rotate_mol_to_symels(mol, paxis, saxis)
        ctab = pg_to_chartab(pg)
        class_map = generate_symel_to_class_map(symels, ctab)
        atom_map = get_atom_mapping(mol, symels)
        mult_table = build_mult_table(symels)
        return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, ctab, class_map, atom_map, mult_table)

    @classmethod
    def from_file(cls, fn):
        with open(fn, "r") as lfn:
            strang = lfn.read()
        schema = qcel.models.Molecule.from_data(strang).dict()
        mol = Molecule.from_schema(schema)
        return cls.from_molecule(mol)
    
    @property
    def rotational_symmetry_number(self):
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

    def subgroup_symtext(self, subgroup):
        subgroup_symels = main.pg_to_symels(subgroup)
        subgroup_ctab = main.pg_to_chartab(subgroup)
        class_map = generate_symel_to_class_map(subgroup_symels, subgroup_ctab)
        mult_table = build_mult_table(subgroup_symels)
        isomorphism = subgroup_by_name(self.symels, self.mult_table, subgroup)
        if isomorphism is None:
            raise Exception(f"No {subgroup} subgroup found for {self.pg} group")
        sgp = [self.symels[i[1]] for i in isomorphism]
        paxis, saxis = subgroup_axes(subgroup, sgp)
        new_mol, reverse_rotate, rotate_to_std = main.rotate_mol_to_symels(self.mol, paxis, saxis)
        new_mol.tol = 1e-10
        atom_map = get_atom_mapping(new_mol, subgroup_symels)
        return Symtext(new_mol, rotate_to_std, reverse_rotate, subgroup, subgroup_symels, subgroup_ctab, class_map, atom_map, mult_table)
    
    def largest_D2h_subgroup(self):
        # Some groups may have equivalent order subgroups, if you want a specific one, don't use this function
        D2h_subgroups = ["D2h", "D2", "C2v", "C2h", "Cs", "C2", "Ci", "C1"]
        for i in D2h_subgroups:
            try:
                return self.subgroup_symtext(i)
            except:
                pass
