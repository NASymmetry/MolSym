import pytest
import numpy as np
import molsym

# A subgroup Symtext built from a standard-frame parent must still rotate back to the
# molecule's original frame. The parent's own rotation to its standard frame has to be
# folded into the subgroup's reverse_rotate, or nonstandard_symtext lands in the
# parent's standard frame instead.

WATER = (["O", "H", "H"], [15.995, 1.008, 1.008])
_ON_Z = np.array([[0, 0, -0.0656], [0, 0.7572, 0.5205], [0, -0.7572, 0.5205]])
_c, _s = np.cos(0.7), np.sin(0.7)
_TILT = np.array([[1, 0, 0], [0, _c, -_s], [0, _s, _c]]) @ np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
WATER_FRAMES = {
    "on z": _ON_Z,
    "tilted": _ON_Z @ _TILT.T,
    "offset": _ON_Z + np.array([-0.7, 0.3, 1.1]),
}


@pytest.mark.parametrize("frame", WATER_FRAMES)
@pytest.mark.parametrize("subgroup", ["C1", "C2", "Cs"])
def test_subgroup_nonstandard_frame_matches_original_geometry(frame, subgroup):
    atoms, masses = WATER
    mol = molsym.Molecule(atoms, np.array(WATER_FRAMES[frame], dtype=float), masses)
    reference = molsym.Molecule(atoms, np.array(WATER_FRAMES[frame], dtype=float), masses)
    reference.translate(reference.find_com())

    parent = molsym.Symtext.from_molecule(mol)
    assert parent.pg.str == "C2v"
    sub = molsym.Symtext.nonstandard_symtext(parent.subgroup_symtext(subgroup))
    assert sub.pg.str == subgroup
    np.testing.assert_allclose(sub.mol.coords, reference.coords, atol=1e-6)
    np.testing.assert_allclose(sub.rotate_to_std @ sub.reverse_rotate, np.eye(3), atol=1e-12)
