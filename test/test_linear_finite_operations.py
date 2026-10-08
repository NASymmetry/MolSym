import pytest
import numpy as np
import molsym

from molsym.salcs.cartesian_coordinates import LinearCartesian
from molsym.salcs.projection_op import ProjectionOp
from molsym.salcs.salc_tools import finite_operations, generate_symmetric_partner, maps_to_negative

# Linear symmetry elements carry no Cartesian matrices, so maps_to_negative and
# generate_symmetric_partner work from the C2v / D2h subgroup about the molecular axis.
LINEAR_MOLECULES = {
    "CO2": (["C", "O", "O"], [[0, 0, 0], [0, 0, 1.17], [0, 0, -1.17]], [12.0, 15.995, 15.995]),
    "HCN": (["H", "C", "N"], [[0, 0, -1.065], [0, 0, 0], [0, 0, 1.153]], [1.008, 12.0, 14.003]),
}
# irrep -> do displacements map to their negative
NEGATES = {
    "CO2": {"Sigma_g^+": False, "Sigma_u^+": True, "Pi_u": True},
    "HCN": {"Sigma^+": False, "Pi": True},
}


def _linear_salcs(label):
    atoms, coords, masses = LINEAR_MOLECULES[label]
    symtext = molsym.Symtext.from_molecule(molsym.Molecule(atoms, np.array(coords, dtype=float), masses))
    salcs = ProjectionOp(symtext, LinearCartesian(symtext), project_Eckart="both")
    salcs.sort_to("blocks")
    return symtext, salcs


def test_linear_finite_operations_atom_maps():
    # D2h about the CO2 axis: E, C2(z), sigma, sigma keep the oxygens; i, sigma_h, C2', C2' swap them
    symtext, _ = _linear_salcs("CO2")
    maps = [op_map.tolist() for _, op_map in finite_operations(symtext)]
    assert maps == [[0, 1, 2]] * 4 + [[0, 2, 1]] * 4


@pytest.mark.parametrize("label", LINEAR_MOLECULES)
def test_linear_maps_to_negative(label):
    symtext, salcs = _linear_salcs(label)
    for h, indices in enumerate(salcs.salcs_by_irrep):
        for i in indices:
            assert maps_to_negative(symtext, salcs[i]) == NEGATES[label][symtext.irreps[h].symbol]


@pytest.mark.parametrize("label", LINEAR_MOLECULES)
def test_linear_symmetric_partner_is_negated_displacement(label):
    symtext, salcs = _linear_salcs(label)
    for salc in salcs:
        partner, op = generate_symmetric_partner(symtext, salc, salc.coeffs.reshape(-1, 3), data_type="gradient")
        if maps_to_negative(symtext, salc):
            assert np.allclose(partner.flatten(), -salc.coeffs)
        else:
            assert partner is None and op is None
