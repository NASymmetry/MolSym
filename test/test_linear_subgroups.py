import pytest
import numpy as np
import molsym

from molsym.salcs.cartesian_coordinates import CartesianCoordinates
from molsym.salcs.projection_op import ProjectionOp

# Finite subgroups of linear point groups are oriented from the molecule: the principal
# axis is the molecular axis (Cs: the mirror contains it), the secondary axis any
# perpendicular direction.

_c, _s = np.cos(0.7), np.sin(0.7)
_TILT = np.array([[1, 0, 0], [0, _c, -_s], [0, _s, _c]]) @ np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])

LINEAR = {
    "CO2": (["C", "O", "O"], np.array([[0, 0, 0], [0, 0, 1.17], [0, 0, -1.17]]), [12.0, 15.995, 15.995]),
    "HCN": (["H", "C", "N"], np.array([[0, 0, -1.065], [0, 0, 0], [0, 0, 1.153]]), [1.008, 12.0, 14.003]),
}
SUBGROUPS = {
    "CO2": ["C1", "Ci", "Cs", "C2", "C2v", "C2h", "D2", "D2h", "C3v", "C4v", "D4h", "D3d", "S4"],
    "HCN": ["C1", "Cs", "C2", "C2v", "C3v", "C4v"],
}
NOT_SUBGROUPS = {"HCN": ["Ci", "C2h", "D2", "D2h", "D4h"]}


def _symtext(label, frame=None):
    atoms, coords, masses = LINEAR[label]
    coords = coords if frame is None else coords @ frame.T
    return molsym.Symtext.from_molecule(molsym.Molecule(atoms, np.array(coords, dtype=float), masses))


@pytest.mark.parametrize("label, subgroup", [(l, s) for l in SUBGROUPS for s in SUBGROUPS[l]])
def test_linear_subgroup_symtext(label, subgroup):
    parent = _symtext(label)
    assert parent.pg.is_linear
    sub = parent.subgroup_symtext(subgroup)
    assert sub.pg.str == subgroup
    assert not sub.pg.is_linear
    assert sub.atom_map.shape == (len(LINEAR[label][0]), len(sub.symels))


@pytest.mark.parametrize("label, subgroup", [(l, s) for l in NOT_SUBGROUPS for s in NOT_SUBGROUPS[l]])
def test_linear_subgroup_symtext_rejects_missing_symmetry(label, subgroup):
    with pytest.raises(Exception):
        _symtext(label).subgroup_symtext(subgroup)


@pytest.mark.parametrize("label, subgroup", [(l, s) for l in SUBGROUPS for s in SUBGROUPS[l]])
def test_linear_subgroup_nonstandard_frame_and_salcs(label, subgroup):
    # Tilted molecule, through the same path psi4 takes: subgroup -> nonstandard -> Cartesian SALCs
    atoms, coords, masses = LINEAR[label]
    reference = molsym.Molecule(atoms, np.array(coords @ _TILT.T, dtype=float), masses)
    reference.translate(reference.find_com())

    sub = molsym.Symtext.nonstandard_symtext(_symtext(label, _TILT).subgroup_symtext(subgroup))
    np.testing.assert_allclose(sub.mol.coords, reference.coords, atol=1e-6)

    salcs = ProjectionOp(sub, CartesianCoordinates(sub), project_Eckart="both")
    assert len(salcs) == 3 * len(atoms) - 5
