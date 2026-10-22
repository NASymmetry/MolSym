import pytest
import numpy as np
import molsym

from molsym.salcs.cartesian_coordinates import LinearCartesian
from molsym.salcs.projection_op import ProjectionOp
from molsym.salcs.salc_tools import generate_degenerate_partner, linear_axis, maps_to_negative

# LinearCartesian builds its SALCs with the molecular axis as z, then rotates them into the
# molecule's own frame, so a linear molecule along any axis gets the on-z SALCs rotated with it.

LINEAR = {
    "CO2": (["C", "O", "O"], np.array([[0, 0, 0], [0, 0, 1.17], [0, 0, -1.17]]), [12.0, 15.995, 15.995],
            {"Sigma_g^+": False, "Sigma_u^+": True, "Pi_u": True}, "Pi_u"),
    "HCN": (["H", "C", "N"], np.array([[0, 0, -1.065], [0, 0, 0], [0, 0, 1.153]]), [1.008, 12.0, 14.003],
            {"Sigma^+": False, "Pi": True}, "Pi"),
}
AXES = [[1, 0, 0], [0, -1, 0], [1, 1, 1], [0.3, -0.5, 0.8]]


def _rotation_z_to(axis):
    axis = np.array(axis, dtype=float) / np.linalg.norm(axis)
    v = np.cross([0, 0, 1], axis)
    if np.linalg.norm(v) < 1e-12:
        return np.diag([1.0, -1.0, -1.0]) if axis[2] < 0 else np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + axis[2])


def _salcs(label, R=np.eye(3)):
    atoms, coords, masses, _, _ = LINEAR[label]
    symtext = molsym.Symtext.from_molecule(molsym.Molecule(atoms, coords @ R.T, masses))
    salcs = ProjectionOp(symtext, LinearCartesian(symtext), project_Eckart="both")
    salcs.sort_to("blocks")
    return symtext, salcs


@pytest.mark.parametrize("label", LINEAR)
@pytest.mark.parametrize("axis", AXES)
def test_off_axis_salcs_are_rotated_on_z_salcs(label, axis):
    R = _rotation_z_to(axis)
    ref_symtext, ref_salcs = _salcs(label)
    symtext, salcs = _salcs(label, R)
    unit = np.array(axis, dtype=float) / np.linalg.norm(axis)
    assert np.allclose(np.abs(linear_axis(symtext) @ unit), 1.0)
    negates = LINEAR[label][3]
    for h, (ref_idx, idx) in enumerate(zip(ref_salcs.salcs_by_irrep, salcs.salcs_by_irrep)):
        assert len(ref_idx) == len(idx)
        if not idx:
            continue
        ref_rot = np.array([(ref_salcs[i].coeffs.reshape(-1, 3) @ R.T).flatten() for i in ref_idx])
        new = np.array([salcs[i].coeffs for i in idx])
        # same span per irrep (partners may mix): projecting the rotated reference loses nothing
        assert np.allclose(ref_rot @ new.T @ new, ref_rot, atol=1e-8)
        for i in idx:
            assert maps_to_negative(symtext, salcs[i]) == negates[symtext.irreps[h].symbol]


@pytest.mark.parametrize("label", LINEAR)
@pytest.mark.parametrize("axis", AXES)
def test_off_axis_degenerate_partner(label, axis):
    symtext, salcs = _salcs(label, _rotation_z_to(axis))
    pi = LINEAR[label][4]
    idx = salcs.salcs_by_irrep[[ir.symbol for ir in symtext.irreps].index(pi)]
    half = len(idx) // 2
    for k in range(half):
        first, second = salcs[idx[k]], salcs[idx[k + half]]
        partner = generate_degenerate_partner(symtext, first, second, first.coeffs.reshape(-1, 3), data_type="gradient")
        np.testing.assert_allclose(partner.flatten(), second.coeffs, atol=1e-10)


def test_dipole_components_follow_the_molecular_axis():
    atoms, coords, masses, _, _ = LINEAR["CO2"]
    along_x = molsym.Symtext.from_molecule(molsym.Molecule(atoms, coords[:, [2, 0, 1]], masses))
    assert along_x.assign_dipole_irrep["Sigma_u^+"] == [(0, 0)]
    assert along_x.assign_dipole_irrep["Pi_u"] == [(1, 0), (2, 1)]
    tilted = molsym.Symtext.from_molecule(molsym.Molecule(atoms, coords @ _rotation_z_to([1, 1, 1]).T, masses))
    assert not any(tilted.assign_dipole_irrep.values())
