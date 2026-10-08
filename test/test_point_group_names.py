import pytest
import numpy as np
import molsym

from molsym.symtext.point_group import PointGroup


@pytest.mark.parametrize("name, expected", [
    ("C2v", "C2v"), ("c2v", "C2v"), ("C2V", "C2v"),
    ("cs", "Cs"), ("CI", "Ci"), ("c1", "C1"), ("d2h", "D2h"),
    ("TD", "Td"), ("s4", "S4"), ("d0h", "D0h"), ("KH", "Kh"),
])
def test_from_string_any_capitalization(name, expected):
    pg = PointGroup.from_string(name)
    ref = PointGroup.from_string(expected)
    assert pg.str == expected
    assert (pg.family, pg.n, pg.subfamily, pg.is_linear) == (ref.family, ref.n, ref.subfamily, ref.is_linear)


WATER = (["O", "H", "H"], np.array([[0.0, 0.0, -0.1287], [0.0, 1.4307, 1.0213], [0.0, -1.4307, 1.0213]]),
         [15.995, 1.008, 1.008])


def test_subgroup_symtext_lowercase_name():
    symtext = molsym.Symtext.from_molecule(molsym.Molecule(*WATER))
    assert symtext.subgroup_symtext("c1").pg.str == "C1"
    assert symtext.subgroup_symtext("cs").pg.str == "Cs"
