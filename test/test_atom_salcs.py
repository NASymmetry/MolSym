import pytest
import numpy as np
import molsym

from molsym.salcs.cartesian_coordinates import CartesianCoordinates
from molsym.salcs.projection_op import ProjectionOp
from molsym.salcs.salc_tools import maps_to_negative

# A single atom (Kh): its x, y, z displacements are the three components of P,
# and they are pure translations.


def _atom_symtext(xyz=(0.0, 0.0, 0.0)):
    return molsym.Symtext.from_molecule(molsym.Molecule(["He"], np.array([xyz]), [4.0026]))


def test_atom_point_group():
    symtext = _atom_symtext()
    assert symtext.pg.family == "K"
    assert symtext.assign_dipole_irrep["P"] == [(0, 0), (1, 1), (2, 2)]


@pytest.mark.parametrize("project_Eckart", ["both", "translational"])
def test_atom_salcs_translations_projected(project_Eckart):
    symtext = _atom_symtext()
    salcs = ProjectionOp(symtext, CartesianCoordinates(symtext), project_Eckart=project_Eckart)
    salcs.sort_to("blocks")
    assert len(salcs) == 0
    assert all(len(idx) == 0 for idx in salcs.salcs_by_irrep)


@pytest.mark.parametrize("project_Eckart", ["rotational", None])
def test_atom_salcs_translations_kept(project_Eckart):
    symtext = _atom_symtext()
    salcs = ProjectionOp(symtext, CartesianCoordinates(symtext), project_Eckart=project_Eckart)
    salcs.sort_to("blocks")
    p_idx = [irrep.symbol for irrep in symtext.irreps].index("P")
    assert salcs.salcs_by_irrep[p_idx] == [0, 1, 2]
    assert np.allclose(salcs.basis_transformation_matrix, np.eye(3))
    assert [salc.i for salc in salcs] == [0, 1, 2]
    # all three are partners of one another
    assert salcs.sort_partner_functions() == [[0, 1, 2]]
    # an atom has no operation that negates a displacement
    assert not any(maps_to_negative(symtext, salc) for salc in salcs)


def test_atom_nonstandard_symtext():
    symtext = _atom_symtext((0.3, -1.2, 2.0))
    nonstandard = molsym.Symtext.nonstandard_symtext(symtext)
    assert nonstandard.is_nonstandard
    assert nonstandard.pg.family == "K"
    salcs = ProjectionOp(nonstandard, CartesianCoordinates(nonstandard), project_Eckart=None)
    assert len(salcs) == 3


def test_atom_rejects_other_function_sets():
    symtext = _atom_symtext()
    with pytest.raises(NotImplementedError):
        ProjectionOp(symtext, object(), project_Eckart=None)
