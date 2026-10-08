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


# Water from psi4's fd-freq-energy test, in psi4's frame (yz plane). Re-deriving C2v's own axes
# here swaps which mirror is sigma_v(0), so asking for the group a Symtext already has must
# return it unchanged.
WATER = (["O", "H", "H"], np.array([[0.0, 0.0, -0.1287], [0.0, 1.4307, 1.0213], [0.0, -1.4307, 1.0213]]),
         [15.995, 1.008, 1.008])


@pytest.mark.parametrize("name", ["C2v", "c2v"])
@pytest.mark.parametrize("nonstandard", [False, True])
def test_subgroup_symtext_of_own_group_is_unchanged(name, nonstandard):
    symtext = molsym.Symtext.from_molecule(molsym.Molecule(*WATER))
    if nonstandard:
        symtext = molsym.Symtext.nonstandard_symtext(symtext)
    same = symtext.subgroup_symtext(name)
    assert same is not symtext
    assert same.pg.str == symtext.pg.str
    assert same.is_nonstandard == symtext.is_nonstandard
    np.testing.assert_allclose(same.mol.coords, symtext.mol.coords)
    np.testing.assert_array_equal(same.atom_map, symtext.atom_map)
    for a, b in zip(same.symels, symtext.symels):
        assert a.symbol == b.symbol
        np.testing.assert_allclose(a.rrep, b.rrep)


def test_subgroup_symtext_lowercase_name():
    symtext = molsym.Symtext.from_molecule(molsym.Molecule(*WATER))
    assert symtext.subgroup_symtext("c1").pg.str == "C1"
    assert symtext.subgroup_symtext("cs").pg.str == "Cs"
