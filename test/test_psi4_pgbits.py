"""
    Psi4's symmetry algorithms have been plagued by the infamous
    "Unrecognized point group bits" error. Hopefully we do not
    also fail for some of the reported molecules.
"""

import pytest
import os
import numpy as np
import molsym

PATH = os.path.dirname(os.path.realpath(__file__))
fns = ["f1.xyz", "f2.xyz", "f3.xyz", "f4.xyz", "f5.xyz", "f6.xyz"]
answers = ["D0h", "Td", "C2v", "C2", "Td", "Td"]

@pytest.mark.parametrize("fn, answer", [(fns[i], answers[i]) for i in range(len(fns))])
def test_unrecognized_point_group_bits(fn, answer):
    # For now, just check point group detection algorithm
    file_path = os.path.join(PATH, "point_group_bits", fn)
    mol = molsym.Molecule.from_file(file_path)
    pg, (paxis, saxis) = molsym.find_point_group(mol)
    assert pg == answer