import pytest
from molsym.symtext.point_group import PointGroup
from molsym.symtext.symtext import Symtext
from molsym.salcs.internal_coordinates import user_to_IC, Stretch, Bend, Torsion, OutOfPlane, InternalCoordinates

user_input = [
    [[0,1], "R1"],
    [[0,1,2], "A23"],
    [[0,1,2,3], "D932j"],
    [[0,1,2,3], "Ode3"] # Random names should be ignored other than leading char
]

r = Stretch([0,1])
a = Bend([0,1,2])
d = Torsion([0,1,2,3])
o = OutOfPlane([0,1,2,3])

allics = [r, a, d, o]
scrambled = [
    Stretch([1,0]),
    Bend([2,1,0]),
    Torsion([3,2,1,0]),
    OutOfPlane([0,1,3,2])
]

def test_Stretch():
    assert r.atom_list == [0,1]
    assert r.phase_on_permute == 1
    assert r.perm_symmetry == r.reversal

def test_Bend():
    assert a.atom_list == [0,1,2]
    assert a.phase_on_permute == 1
    assert a.perm_symmetry == a.reversal

def test_Torsion():
    assert d.atom_list == [0,1,2,3]
    assert d.phase_on_permute == 1
    assert d.perm_symmetry == d.reversal

def test_OutOfPlane():
    assert o.atom_list == [0,1,2,3]
    assert o.phase_on_permute == -1
    assert o.exchange_atoms == [0,1,3,2]
    assert o.perm_symmetry == o.exchange

def test_convert_internal_coords():
    out = [user_to_IC(k) for k in user_input]
    for i in range(len(out)):
        print(out[i])
        assert out[i] == allics[i]

@pytest.mark.parametrize("i", [i for i in range(len(scrambled))])
def test_find_equiv_ic(i):
    symtext = Symtext.empty()
    IC_fxn_set = InternalCoordinates(symtext, user_input)
    out = IC_fxn_set.find_equiv_ic(scrambled[i])
    assert out[0] == i
    assert out[1] == scrambled[i].phase_on_permute
