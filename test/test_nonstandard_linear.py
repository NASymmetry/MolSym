import pytest
import numpy as np
import molsym

from molsym.salcs.cartesian_coordinates import LinearCartesian
from molsym.salcs.projection_op import ProjectionOp

# Symtext.from_molecule orients a linear molecule along z; nonstandard_symtext moves it back
# to the input frame (linear symmetry elements carry no matrices, so only the molecule moves).

LINEAR = {
    "CO2": (["C", "O", "O"], np.array([[0, 0, 0], [0, 0, 1.17], [0, 0, -1.17]], dtype=float), [12.0, 15.995, 15.995]),
    "HCN": (["H", "C", "N"], np.array([[0, 0, -1.065], [0, 0, 0], [0, 0, 1.153]], dtype=float), [1.008, 12.0, 14.003]),
}
_c, _s = np.cos(0.7), np.sin(0.7)
_TILT = np.array([[1, 0, 0], [0, _c, -_s], [0, _s, _c]]) @ np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
FRAMES = {
    "on z": np.eye(3),
    "along x": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
    "tilted": _TILT,
}


def _input(label, frame, offset=(0.0, 0.0, 0.0)):
    atoms, coords, masses = LINEAR[label]
    return molsym.Molecule(atoms, coords @ FRAMES[frame].T + np.array(offset), masses)


@pytest.mark.parametrize("label", LINEAR)
@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (0.4, -1.1, 2.3)])
def test_nonstandard_linear_returns_input_frame(label, frame, offset):
    reference = _input(label, frame, offset)
    reference.translate(reference.find_com())
    symtext = molsym.Symtext.from_molecule(_input(label, frame, offset))
    nonstandard = molsym.Symtext.nonstandard_symtext(symtext)
    assert nonstandard.is_nonstandard and nonstandard.pg.is_linear
    assert nonstandard.pg.str == symtext.pg.str
    np.testing.assert_allclose(nonstandard.mol.coords, reference.coords, atol=1e-10)
    np.testing.assert_array_equal(nonstandard.atom_map, symtext.atom_map)


@pytest.mark.parametrize("label", LINEAR)
@pytest.mark.parametrize("frame", FRAMES)
def test_nonstandard_linear_salcs_are_rotated_standard_salcs(label, frame):
    symtext = molsym.Symtext.from_molecule(_input(label, frame))
    nonstandard = molsym.Symtext.nonstandard_symtext(symtext)
    std = ProjectionOp(symtext, LinearCartesian(symtext))
    new = ProjectionOp(nonstandard, LinearCartesian(nonstandard))
    std.sort_to("blocks"); new.sort_to("blocks")
    Q = symtext.reverse_rotate
    for h, (std_idx, new_idx) in enumerate(zip(std.salcs_by_irrep, new.salcs_by_irrep)):
        assert len(std_idx) == len(new_idx)
        if not new_idx:
            continue
        rotated = np.array([(std[i].coeffs.reshape(-1, 3) @ Q.T).flatten() for i in std_idx])
        coeffs = np.array([new[i].coeffs for i in new_idx])
        # same span per irrep (partners may mix)
        np.testing.assert_allclose(rotated @ coeffs.T @ coeffs, rotated, atol=1e-8)


def test_nonstandard_linear_dipole_assignment_follows_input_axis():
    along_x = molsym.Symtext.nonstandard_symtext(molsym.Symtext.from_molecule(_input("CO2", "along x")))
    assert along_x.assign_dipole_irrep["Sigma_u^+"] == [(0, 0)]
    assert along_x.assign_dipole_irrep["Pi_u"] == [(1, 0), (2, 1)]
    tilted = molsym.Symtext.nonstandard_symtext(molsym.Symtext.from_molecule(_input("CO2", "tilted")))
    assert not any(tilted.assign_dipole_irrep.values())


def test_nonstandard_linear_rejects_already_nonstandard():
    nonstandard = molsym.Symtext.nonstandard_symtext(molsym.Symtext.from_molecule(_input("HCN", "along x")))
    with pytest.raises(ValueError):
        molsym.Symtext.nonstandard_symtext(nonstandard)
