import os
from molsym.flowchart import find_point_group
#import psi4
import qcelemental as qcel
from molsym.molecule import Molecule
import pytest

PATH = os.path.dirname(os.path.realpath(__file__))

names = ["C1", "Ci", "Cs", "Cs_nonplanar", "C2", "C3", "S4", "C2v", "C3v", "C2h", "C3h", "D2", "D3", "D2d", "D4d", "D3h", "D4h", "D5h", "D6h", "D12h", "D100h", 
         "Cinfv", "Dinfh", "Th", "Td", "cube", "octahedron", "dodecahedron", "icosahedron"]
pgs = ["C1", "Ci", "Cs", "Cs", "C2", "C3", "S4", "C2v", "C3v", "C2h", "C3h", "D2", "D3", "D2d", "D4d", "D3h", "D4h", "D5h", "D6h", "D12h", "D100h", 
       "C0v", "D0h", "Th", "Td", "Oh", "Oh", "Ih", "Ih"]

#files = ["benzene.xyz", "C1.xyz", "C2.xyz", "C2h.xyz", "C3v.xyz", "D2d.xyz", "D4d.xyz", "D3.xyz", "D3h.xyz", "D6h.xyz"]
#pgs = ["D6h", "C1", "C2", "C2h", "C3v", "D2d", "D4d", "D3", "D3h", "D6h"]

@pytest.mark.parametrize("name, pg_ans", [(names[i], pgs[i]) for i in range(len(names))])
def test_find_point_group(name, pg_ans):
    file_path = os.path.join(PATH, "sxyz", f"{name}.xyz")
    with open(file_path, "r") as fn:
        strang = fn.read()
    schema = qcel.models.Molecule.from_data(strang).dict()
    mol = Molecule.from_schema(schema)
    pg, (paxis, saxis) = find_point_group(mol)
    print("Ans: ", pg)
    assert pg_ans == pg
