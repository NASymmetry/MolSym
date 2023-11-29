from molsym.molecule import *
from molsym import find_point_group
from .main import *
from .multiplication_table import build_mult_table

class Symtext():
    def __init__(self, mol, rotate_to_std, reverse_rotate, pg, symels, chartable, class_map, atom_map, mult_table) -> None:
        self.mol = mol
        self.rotate_to_std = rotate_to_std
        self.reverse_rotate = reverse_rotate
        self.pg = PointGroup.from_string(pg) # TODO TODO TODO I CHANGED THIS AND IT MIGHT BREAK STUFF TODO TODO TODO
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
        mtable = build_mult_table(symels)
        return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, ctab, class_map, atom_map, mtable)

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
