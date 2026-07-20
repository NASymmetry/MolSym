import numpy as np
import pytest
import molsym

from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent

from molsym.salcs.axial_vector_functions import AxialVectorFunctions
from molsym.salcs.projection_op import ProjectionOp


MOLECULES = ["water", "ammonia", "methane"]
REFERENCE_REDUCTIONS = {
    "water": {"A_2": 1, "B_1": 1, "B_2": 1},   
    "ammonia": {"A_2": 1, "E": 1},             
    "methane": {"T_1": 1},                     
}

@pytest.mark.parametrize("molecule", MOLECULES)
def test_axial_vector_fxn_map_is_det_times_rrep(molecule):
    mol = molsym.Molecule.from_file(str(TEST_DIR / "xyz" / f"{molecule}.xyz"))
    symtext = molsym.Symtext.from_molecule(mol)
    fxn_set = AxialVectorFunctions(symtext)

    for sidx, symel in enumerate(symtext.symels):
        A = np.array(symel.rrep, dtype=float)
        det = np.linalg.det(A)

        if abs(det - 1.0) < symtext.mol.tol:
            det = 1.0
        elif abs(det + 1.0) < symtext.mol.tol:
            det = -1.0

        expected = (det * A).T

        assert np.allclose(fxn_set.fxn_map[sidx],expected,atol=symtext.mol.tol)


@pytest.mark.parametrize("molecule", MOLECULES)
def test_axial_vector_reduction_matches_salcs(molecule):
    mol = molsym.Molecule.from_file(str(TEST_DIR / "xyz" / f"{molecule}.xyz"))
    symtext = molsym.Symtext.from_molecule(mol)
    fxn_set = AxialVectorFunctions(symtext)

    rep_chars_by_operation = np.array([np.trace(fxn_set.fxn_map[sidx].T) for sidx in range(len(symtext))])

    expected_coeffs = symtext.reduction_coefficients(rep_chars_by_operation,by_class=False)

    salcs = ProjectionOp(symtext,fxn_set,project_Eckart=False)

    actual_coeffs = np.zeros(len(symtext.irreps), dtype=int)

    for irrep_idx, irrep in enumerate(symtext.irreps):
        n_salcs = sum(salc.irrep.symbol == irrep.symbol for salc in salcs.salcs)

        assert n_salcs % irrep.d == 0
        actual_coeffs[irrep_idx] = n_salcs // irrep.d

    assert np.array_equal(actual_coeffs, expected_coeffs)


@pytest.mark.parametrize("molecule", MOLECULES)
def test_axial_vector_salcs_match_character_projectors(molecule):
    mol = molsym.Molecule.from_file(str(TEST_DIR / "xyz" / f"{molecule}.xyz"))
    symtext = molsym.Symtext.from_molecule(mol)
    fxn_set = AxialVectorFunctions(symtext)

    salcs = ProjectionOp(symtext,fxn_set,project_Eckart=False)

    T_mats = np.array([fxn_set.fxn_map[sidx].T for sidx in range(len(symtext))])

    for irrep_idx, irrep in enumerate(symtext.irreps):
        matching = [salc.coeffs for salc in salcs.salcs if salc.irrep.symbol == irrep.symbol]

        if len(matching) == 0:
            continue

        B = np.column_stack(matching)
        P_salc = B @ B.T

        P_char = np.zeros((3, 3))

        for sidx in range(len(symtext)):
            chi = np.trace(symtext.irrep_mats[irrep.symbol][sidx])
            P_char += chi * T_mats[sidx]

        P_char *= irrep.d / symtext.order

        assert np.allclose(P_salc,P_char,atol=symtext.mol.tol)


@pytest.mark.parametrize("molecule", MOLECULES)
def test_axial_vector_known_reductions(molecule):
    mol = molsym.Molecule.from_file(str(TEST_DIR / "xyz" / f"{molecule}.xyz"))
    symtext = molsym.Symtext.from_molecule(mol)

    fxn_set = AxialVectorFunctions(symtext)
    salcs = ProjectionOp(symtext, fxn_set, project_Eckart=False)

    actual = {}

    for irrep in symtext.irreps:
        n_salcs = sum(salc.irrep.symbol == irrep.symbol for salc in salcs.salcs)

        mult = n_salcs // irrep.d

        if mult:
            actual[irrep.symbol] = mult

    assert actual == REFERENCE_REDUCTIONS[molecule]
