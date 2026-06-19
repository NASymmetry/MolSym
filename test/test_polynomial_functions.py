import numpy as np
import pytest
import molsym

from pathlib import Path

from molsym.salcs.polynomial_functions import PolynomialFunctions
from molsym.salcs.projection_op import ProjectionOp


TEST_DIR = Path(__file__).resolve().parent
MOLECULES = ["water", "ammonia", "methane"]


REFERENCE_REDUCTIONS = {
    1: {
        "water": {"A_1": 1, "B_1": 1, "B_2": 1},
        "ammonia": {"A_1": 1, "E": 1},
        "methane": {"T_2": 1},
    },
    2: {
        "water": {"A_1": 3, "A_2": 1, "B_1": 1, "B_2": 1},
        "ammonia": {"A_1": 2, "E": 2},
        "methane": {"A_1": 1, "E": 1, "T_2": 1},
    },
    3: {
        "water": {"A_1": 3, "A_2": 1, "B_1": 3, "B_2": 3},
        "ammonia": {"A_1": 3, "A_2": 1, "E": 3},
        "methane": {"A_1": 1, "T_1": 1, "T_2": 2},
    },
}


REFERENCE_FUNCTIONS = {
    1: {
        # basis: x, y, z
        "water": {
            "A_1": [[0, 0, 1]],
            "B_1": [[1, 0, 0]],
            "B_2": [[0, 1, 0]],
        },
        "ammonia": {
            "A_1": [[0, 0, 1]],
            "E": [
                [1, 0, 0],
                [0, 1, 0],
            ],
        },
        "methane": {
            "T_2": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        },
    },

    2: {
        # basis: x², xy, xz, y², yz, z²
        "water": {
            "A_1": [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            "A_2": [[0, 1, 0, 0, 0, 0]],
            "B_1": [[0, 0, 1, 0, 0, 0]],
            "B_2": [[0, 0, 0, 0, 1, 0]],
        },
        "ammonia": {
            "A_1": [
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            "E": [
                [1, 0, 0, -1, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
        },
        "methane": {
            "A_1": [[1, 0, 0, 1, 0, 1]],
            "E": [
                [-1, 0, 0, -1, 0, 2],
                [1, 0, 0, -1, 0, 0],
            ],
            "T_2": [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
        },
    },

    3: {
        # basis: x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³
        "water": {
            "A_1": [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # x²z
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # y²z
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # z³
            ],
            "A_2": [
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # xyz
            ],
            "B_1": [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x³
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # xy²
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # xz²
            ],
            "B_2": [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # x²y
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # y³
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # yz²
            ],
        },
        "ammonia": {
            "A_1": [
                [1, 0, 0, -3, 0, 0, 0, 0, 0, 0],  # x³ - 3xy²
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],   # x²z + y²z
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # z³
            ],
            "A_2": [
                [0, 3, 0, 0, 0, 0, -1, 0, 0, 0],  # 3x²y - y³
            ],
            "E": [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # x³ + xy²
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],   # x²y + y³
                [0, 0, 1, 0, 0, 0, 0, -1, 0, 0],  # x²z - y²z
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # xyz
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # xz²
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # yz²
            ],
        },
        "methane": {
            "A_1": [
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # xyz
            ],
            "T_1": [
                [0, 0, 0, 1, 0, -1, 0, 0, 0, 0],  # x(y²-z²)
                [0, -1, 0, 0, 0, 0, 0, 0, 1, 0],  # y(z²-x²)
                [0, 0, 1, 0, 0, 0, 0, -1, 0, 0],  # z(x²-y²)
            ],
            "T_2": [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # x³
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # y³
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # z³
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],   # x(y²+z²)
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],   # y(x²+z²)
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],   # z(x²+y²)
            ],
        },
    },
}

CASES = [
    (molecule, degree)
    for molecule in MOLECULES
    for degree in [1, 2, 3]
]

def load_symtext(molecule):
    xyz = TEST_DIR / "xyz" / f"{molecule}.xyz"
    mol = molsym.Molecule.from_file(str(xyz))
    return molsym.Symtext.from_molecule(mol)


def subspace_projector(vectors):
    B = np.column_stack([np.array(v, dtype=float) for v in vectors])
    Q, _ = np.linalg.qr(B)
    return Q @ Q.T


@pytest.mark.parametrize("molecule, degree", CASES)
def test_polynomial_reductions(molecule, degree):
    symtext = load_symtext(molecule)
    fxn_set = PolynomialFunctions(symtext, degree=degree)
    salcs = ProjectionOp(
        symtext,
        fxn_set,
        project_Eckart=False,
    )

    actual = {}

    for irrep in symtext.irreps:
        n_salcs = sum(
            salc.irrep.symbol == irrep.symbol
            for salc in salcs.salcs
        )

        mult = n_salcs // irrep.d

        if mult:
            actual[irrep.symbol] = mult

    assert actual == REFERENCE_REDUCTIONS[degree][molecule]


@pytest.mark.parametrize("molecule, degree", CASES)
def test_polynomial_salcs_match_reference_subspaces(molecule, degree):
    symtext = load_symtext(molecule)
    fxn_set = PolynomialFunctions(symtext, degree=degree)
    salcs = ProjectionOp(
        symtext,
        fxn_set,
        project_Eckart=False,
    )

    for irrep_symbol, reference_vectors in REFERENCE_FUNCTIONS[degree][molecule].items():
        matching = [
            salc.coeffs
            for salc in salcs.salcs
            if salc.irrep.symbol == irrep_symbol
        ]

        assert matching, f"No SALCs found for {irrep_symbol}"

        P_ref = subspace_projector(reference_vectors)
        P_test = subspace_projector(matching)

        assert np.allclose(
            P_test,
            P_ref,
            atol=10 * symtext.mol.tol,
        )

@pytest.mark.parametrize("molecule, degree", CASES)
def test_polynomial_fxn_map_characters_reduce_correctly(molecule, degree):
    symtext = load_symtext(molecule)
    fxn_set = PolynomialFunctions(symtext, degree=degree)
    op_chars = np.array([
        np.trace(fxn_set.fxn_map[sidx].T)
        for sidx in range(len(symtext))
    ])
    expected_coeffs = symtext.reduction_coefficients(
        op_chars,
        by_class=False,
    )

    salcs = ProjectionOp(
        symtext,
        fxn_set,
        project_Eckart=False,
    )

    actual_coeffs = np.zeros(len(symtext.irreps), dtype=int)
    for irrep_idx, irrep in enumerate(symtext.irreps):
        n_salcs = sum(
            salc.irrep.symbol == irrep.symbol
            for salc in salcs.salcs
        )

        assert n_salcs % irrep.d == 0
        actual_coeffs[irrep_idx] = n_salcs // irrep.d

    assert np.array_equal(actual_coeffs, expected_coeffs)
