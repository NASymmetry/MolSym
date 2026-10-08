import pytest
import numpy as np
import molsym

from molsym.salcs.cartesian_coordinates import CartesianCoordinates, LinearCartesian
from molsym.salcs.projection_op import ProjectionOp
from molsym.salcs.salc_tools import generate_degenerate_partner

LINEAR = {
    "CO2": (["C", "O", "O"], [[0, 0, 0], [0, 0, 1.17], [0, 0, -1.17]], [12.0, 15.995, 15.995], "Pi_u"),
    "HCN": (["H", "C", "N"], [[0, 0, -1.065], [0, 0, 0], [0, 0, 1.153]], [1.008, 12.0, 14.003], "Pi"),
}


def _pi_pair(label):
    atoms, coords, masses, pi = LINEAR[label]
    symtext = molsym.Symtext.from_molecule(molsym.Molecule(atoms, np.array(coords, dtype=float), masses))
    salcs = ProjectionOp(symtext, LinearCartesian(symtext), project_Eckart="both")
    salcs.sort_to("blocks")
    idx = salcs.salcs_by_irrep[[irrep.symbol for irrep in symtext.irreps].index(pi)]
    half = len(idx) // 2
    return symtext, salcs, [(salcs[idx[k]], salcs[idx[k + half]]) for k in range(half)]


@pytest.mark.parametrize("label", LINEAR)
def test_partner_of_displacement_is_partner_salc(label):
    # A displacement transforms like a gradient, so its partner is the partner SALC itself.
    symtext, _, pairs = _pi_pair(label)
    for first, second in pairs:
        partner = generate_degenerate_partner(symtext, first, second, first.coeffs.reshape(-1, 3), data_type="gradient")
        np.testing.assert_allclose(partner.flatten(), second.coeffs, atol=1e-10)


@pytest.mark.parametrize("label", LINEAR)
def test_dipole_partner_is_perpendicular_rotation(label):
    symtext, _, pairs = _pi_pair(label)
    first, second = pairs[0]
    stencil = np.array([[1.0, 0.0, 0.3], [-1.0, 0.0, 0.3], [2.0, 0.0, -0.1], [-2.0, 0.0, -0.1]])
    partner = generate_degenerate_partner(symtext, first, second, stencil, data_type="dipole")
    assert partner.shape == stencil.shape
    # x -> +/- y, and the component along the molecular (z) axis is untouched
    np.testing.assert_allclose(np.abs(partner[:, 1]), np.abs(stencil[:, 0]), atol=1e-12)
    np.testing.assert_allclose(partner[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(partner[:, 2], stencil[:, 2], atol=1e-12)


def test_non_partner_returns_none():
    symtext, salcs, pairs = _pi_pair("CO2")
    sigma = salcs[salcs.salcs_by_irrep[0][0]]
    assert generate_degenerate_partner(symtext, pairs[0][0], sigma, np.eye(3)[0]) is None


def test_nonlinear_raises():
    water = molsym.Molecule(["O", "H", "H"], np.array([[0, 0, -0.0656], [0, 0.7572, 0.5205], [0, -0.7572, 0.5205]]),
                            [15.995, 1.008, 1.008])
    symtext = molsym.Symtext.from_molecule(water)
    salcs = ProjectionOp(symtext, CartesianCoordinates(symtext), project_Eckart="both")
    with pytest.raises(NotImplementedError):
        generate_degenerate_partner(symtext, salcs[0], salcs[1], np.eye(3)[0])
