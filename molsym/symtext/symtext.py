from molsym.molecule import *
from molsym import find_point_group
from .main import *
from .multiplication_table import build_mult_table

class Symtext():
    def __init__(self, mol, pg, symels, chartable, class_map, atom_map, mult_table) -> None:
        self.mol = mol
        self.pg = pg
        self.symels = symels
        self.chartable = chartable
        self.class_map = class_map
        self.atom_map = atom_map
        self.mult_table = mult_table
        self.order = len(symels)

    def __repr__(self):
        return f"\n{self.mol}\n{self.chartable}\n{self.symels}\nClass map:\n{self.class_map}\nAtom map:\n{self.atom_map}\nMultiplication Table\n{self.mult_table}"

    @classmethod
    def from_molecule(cls, mol):
        mol.translate(mol.find_com())
        pg, (paxis, saxis) = find_point_group(mol)
        symels = pg_to_symels(pg)
        mol = rotate_mol_to_symels(mol, paxis, saxis)
        ctab = pg_to_chartab(pg)
        class_map = generate_symel_to_class_map(symels, ctab)
        atom_map = get_atom_mapping(mol, symels)
        mtable = build_mult_table(symels)
        return Symtext(mol, pg, symels, ctab, class_map, atom_map, mtable)

    @classmethod
    def from_file(cls, fn):
        with open(fn, "r") as lfn:
            strang = lfn.read()
        schema = qcel.models.Molecule.from_data(strang).dict()
        mol = Molecule.from_schema(schema)
        return cls.from_molecule(mol)
