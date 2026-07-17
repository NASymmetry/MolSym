import pytest
import numpy as np
import molsym
from pathlib import Path

from molsym.salcs.salc_tools import (
    axial_matrix,
    character_by_operation,
    format_reduction,
    monomial_label,
    prettify_polynomial_string,
    polynomial_salc_to_string,
)

from molsym.salcs.axial_vector_functions import AxialVectorFunctions
from molsym.salcs.polynomial_functions import PolynomialFunctions
from molsym.salcs.internal_coordinates import InternalCoordinates
from molsym.salcs.projection_op import ProjectionOp
from molsym.molecule import global_tol

TEST_DIR = Path(__file__).resolve().parent

reference_data = {

"water": [

{
    "coeffs": np.array([
        -0.0, -0.0,  0.33454,
         0.0,  0.0, -0.66637,
         0.0,  0.0, -0.66637,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
        -0.0,  0.0,  0.0,
         0.0,  0.70711, 0.0,
        -0.0, -0.70711, 0.0,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
        -0.0,  0.27026, -0.0,
        -0.0, -0.53833, -0.41675,
        -0.0, -0.53833,  0.41675,
    ]),
    "maps_to_negative": True,
},

],

"ammonia": [

{
    "coeffs": np.array([
        -0.0, -0.0,  0.42140,
         0.0,  0.0, -0.52359,
         0.0,  0.0, -0.52359,
         0.0,  0.0, -0.52359,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
        -0.0,  0.0, -0.0,
         0.28868,  0.50000,  0.0,
         0.28868, -0.50000,  0.0,
        -0.57735, -0.0,     -0.0,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
         0.37111,  0.0, -0.0,
        -0.46111,  0.0, -0.19339,
        -0.46111,  0.0, -0.19339,
        -0.46111, -0.0,  0.38678,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
         0.0,  0.0, -0.0,
         0.28868, -0.50000, -0.0,
         0.28868,  0.50000,  0.0,
        -0.57735, -0.0,      0.0,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
         0.0,  0.37111, -0.0,
         0.0, -0.46111, -0.33496,
        -0.0, -0.46111,  0.33496,
        -0.0, -0.46111,  0.0,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
         0.0, -0.0,  0.0,
        -0.50000, -0.28868,  0.0,
         0.50000, -0.28868, -0.0,
        -0.0,      0.57735,  0.0,
    ]),
    "maps_to_negative": True,
},

],"methane": [

{
    "coeffs": np.array([
        -0.0,  0.0,  0.0,
         0.28868, -0.28868,  0.28868,
        -0.28868,  0.28868,  0.28868,
         0.28868,  0.28868, -0.28868,
        -0.28868, -0.28868, -0.28868,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
         0.0, -0.0,  0.0,
         0.20412, -0.20412, -0.40825,
        -0.20412,  0.20412, -0.40825,
         0.20412,  0.20412,  0.40825,
        -0.20412, -0.20412,  0.40825,
    ]),
    "maps_to_negative": False,
},

{
    "coeffs": np.array([
        -0.0,  0.0,  0.0,
        -0.35355, -0.35355, -0.0,
         0.35355,  0.35355, -0.0,
        -0.35355,  0.35355,  0.0,
         0.35355, -0.35355,  0.0,
    ]),
    "maps_to_negative": True,
},

{
    "coeffs": np.array([
         0.50146, -0.0,  0.0,
        -0.43259,  0.0, -0.0,
        -0.43259,  0.0, -0.0,
        -0.43259,  0.0, -0.0,
        -0.43259,  0.0, -0.0,
    ]),
    "maps_to_negative": True,
},

{
    "coeffs": np.array([
        -0.0, -0.0, -0.0,
        -0.0,  0.35355, -0.35355,
         0.0,  0.35355,  0.35355,
        -0.0, -0.35355,  0.35355,
        -0.0, -0.35355, -0.35355,
    ]),
    "maps_to_negative": True,
},

{
    "coeffs": np.array([
         0.0,  0.50146,  0.0,
        -0.0, -0.43259,  0.0,
        -0.0, -0.43259,  0.0,
        -0.0, -0.43259,  0.0,
        -0.0, -0.43259,  0.0,
    ]),
    "maps_to_negative": True,
},

{
    "coeffs": np.array([
        -0.0, -0.0,  0.0,
         0.35355,  0.0,  0.35355,
         0.35355,  0.0, -0.35355,
        -0.35355, -0.0,  0.35355,
        -0.35355, -0.0, -0.35355,
    ]),
    "maps_to_negative": True,
},

{
    "coeffs": np.array([
        -0.0,  0.0,  0.50146,
         0.0,  0.0, -0.43259,
         0.0,  0.0, -0.43259,
         0.0, -0.0, -0.43259,
         0.0, -0.0, -0.43259,
    ]),
    "maps_to_negative": True,
},

{
    "coeffs": np.array([
         0.0,  0.0, -0.0,
        -0.35355,  0.35355, -0.0,
         0.35355, -0.35355,  0.0,
         0.35355,  0.35355, -0.0,
        -0.35355, -0.35355,  0.0,
    ]),
    "maps_to_negative": True,
},

],

}

symmetric_partner_reference = {
    "water": [
        {
            "salc_index": 2,
            "salc_coeffs": reference_data["water"][2]["coeffs"],
            "neg_dipole": np.array([0.0, -0.00078, -0.66177]),
            "pos_dipole": np.array([0.0,  0.00078, -0.66177]),
            "neg_gradient": np.array([
                [0.0, -0.00392, -0.01439],
                [0.0, -0.00842,  0.00874],
                [0.0,  0.01233,  0.00565],
            ]),
            "pos_gradient": np.array([
                [ 0.0,  0.00392, -0.01439],
                [-0.0, -0.01233,  0.00565],
                [ 0.0,  0.00842,  0.00874],
            ]),
        },
    ],

    "ammonia": [
        {
            "salc_index": 5,
            "salc_coeffs": reference_data["ammonia"][5]["coeffs"],
            "neg_dipole": np.array([-0.0, -0.00035, -0.71494]),
            "pos_dipole": np.array([-0.0,  0.00035, -0.71494]),
            "neg_gradient": np.array([
                [-0.0,    -0.00189, -0.02752],
                [ 0.00996, 0.01678,  0.00963],
                [ 0.00806,-0.01443,  0.00872],
                [-0.01801,-0.00047,  0.00917],
            ]),
            "pos_gradient": np.array([
                [-0.0,     0.00189, -0.02752],
                [ 0.00806, 0.01443,  0.00872],
                [ 0.00996,-0.01678,  0.00963],
                [-0.01801, 0.00047,  0.00917],
            ]),
        },
    ],

    "methane": [
        {
            "salc_index": 2,
            "salc_coeffs": reference_data["methane"][2]["coeffs"],
            "neg_dipole": np.array([0.0, 0.0, 0.0]),
            "pos_dipole": np.array([0.0, 0.0, 0.0]),
            "neg_gradient": np.array([
                [-0.0,      0.0,      0.0],
                [ 0.00168, -0.00119,  0.00144],
                [-0.00168,  0.00119,  0.00144],
                [ 0.00168,  0.00119, -0.00144],
                [-0.00168, -0.00119, -0.00144],
            ]),
            "pos_gradient": np.array([
                [-0.0,      0.0,      0.0],
                [ 0.00119, -0.00168,  0.00144],
                [-0.00119,  0.00168,  0.00144],
                [ 0.00119,  0.00168, -0.00144],
                [-0.00119, -0.00168, -0.00144],
            ]),
        },
        {
            "salc_index": 3,
            "salc_coeffs": reference_data["methane"][3]["coeffs"],
            "neg_dipole": np.array([0.00024, 0.0, 0.0]),
            "pos_dipole": np.array([-0.00024, 0.0, -0.0]),
            "neg_gradient": np.array([
                [-0.00240,  0.0,      0.0],
                [ 0.00204, -0.00181,  0.00181],
                [-0.00084,  0.00106,  0.00106],
                [ 0.00204,  0.00181, -0.00181],
                [-0.00084, -0.00106, -0.00106],
            ]),
            "pos_gradient": np.array([
                [ 0.00240,  0.0,      0.0],
                [ 0.00084, -0.00106,  0.00106],
                [-0.00204,  0.00181,  0.00181],
                [ 0.00084,  0.00106, -0.00106],
                [-0.00204, -0.00181, -0.00181],
            ]),
        },
        {
            "salc_index": 4,
            "salc_coeffs": reference_data["methane"][4]["coeffs"],
            "neg_dipole": np.array([-0.00035, 0.0, 0.0]),
            "pos_dipole": np.array([0.00035, 0.0, 0.0]),
            "neg_gradient": np.array([
                [-0.00182,  0.0,      0.0],
                [ 0.00189, -0.00209,  0.00209],
                [-0.00098,  0.00079,  0.00079],
                [ 0.00189,  0.00209, -0.00209],
                [-0.00098, -0.00079, -0.00079],
            ]),
            "pos_gradient": np.array([
                [ 0.00182,  0.0,      0.0],
                [ 0.00098, -0.00079,  0.00079],
                [-0.00189,  0.00209,  0.00209],
                [ 0.00098,  0.00079, -0.00079],
                [-0.00189, -0.00209, -0.00209],
            ]),
        },
        {
            "salc_index": 5,
            "salc_coeffs": reference_data["methane"][5]["coeffs"],
            "neg_dipole": np.array([0.0, 0.00024, 0.0]),
            "pos_dipole": np.array([0.0, -0.00024, 0.0]),
            "neg_gradient": np.array([
                [ 0.0,     -0.00240,  0.0],
                [ 0.00106, -0.00084,  0.00106],
                [-0.00181,  0.00204,  0.00181],
                [ 0.00181,  0.00204, -0.00181],
                [-0.00106, -0.00084, -0.00106],
            ]),
            "pos_gradient": np.array([
                [ 0.0,      0.00240, -0.0],
                [ 0.00181, -0.00204,  0.00181],
                [-0.00106,  0.00084,  0.00106],
                [ 0.00106,  0.00084, -0.00106],
                [-0.00181, -0.00204, -0.00181],
            ]),
        },
        {
            "salc_index": 6,
            "salc_coeffs": reference_data["methane"][6]["coeffs"],
            "neg_dipole": np.array([0.0, -0.00035, 0.0]),
            "pos_dipole": np.array([0.0, 0.00035, -0.0]),
            "neg_gradient": np.array([
                [ 0.0,     -0.00182,  0.0],
                [ 0.00079, -0.00098,  0.00079],
                [-0.00209,  0.00189,  0.00209],
                [ 0.00209,  0.00189, -0.00209],
                [-0.00079, -0.00098, -0.00079],
            ]),
            "pos_gradient": np.array([
                [ 0.0,      0.00182, -0.0],
                [ 0.00209, -0.00189,  0.00209],
                [-0.00079,  0.00098,  0.00079],
                [ 0.00079,  0.00098, -0.00079],
                [-0.00209, -0.00189, -0.00209],
            ]),
        },
        {
            "salc_index": 7,
            "salc_coeffs": reference_data["methane"][7]["coeffs"],
            "neg_dipole": np.array([0.0, 0.0, 0.00024]),
            "pos_dipole": np.array([0.0, -0.0, -0.00024]),
            "neg_gradient": np.array([
                [ 0.0,     -0.0,     -0.00240],
                [ 0.00181, -0.00181,  0.00204],
                [-0.00181,  0.00181,  0.00204],
                [ 0.00106,  0.00106, -0.00084],
                [-0.00106, -0.00106, -0.00084],
            ]),
            "pos_gradient": np.array([
                [ 0.0,      0.0,      0.00240],
                [ 0.00106, -0.00106,  0.00084],
                [-0.00106,  0.00106,  0.00084],
                [ 0.00181,  0.00181, -0.00204],
                [-0.00181, -0.00181, -0.00204],
            ]),
        },
        {
            "salc_index": 8,
            "salc_coeffs": reference_data["methane"][8]["coeffs"],
            "neg_dipole": np.array([0.0, 0.0, -0.00035]),
            "pos_dipole": np.array([0.0, 0.0, 0.00035]),
            "neg_gradient": np.array([
                [ 0.0,     -0.0,     -0.00182],
                [ 0.00209, -0.00209,  0.00189],
                [-0.00209,  0.00209,  0.00189],
                [ 0.00079,  0.00079, -0.00098],
                [-0.00079, -0.00079, -0.00098],
            ]),
            "pos_gradient": np.array([
                [ 0.0,      0.0,      0.00182],
                [ 0.00079, -0.00079,  0.00098],
                [-0.00079,  0.00079,  0.00098],
                [ 0.00209,  0.00209, -0.00189],
                [-0.00209, -0.00209, -0.00189],
            ]),
        },
    ],
}

@pytest.mark.parametrize("fn", list(reference_data.keys()))
def test_maps_to_negative(fn):

    mol = molsym.Molecule.from_file(TEST_DIR / "xyz" / f"{fn}.xyz")

    symtext = molsym.Symtext.from_molecule(mol)

    cart_coords = molsym.salcs.CartesianCoordinates(symtext)

    salcs = molsym.salcs.ProjectionOp(symtext, cart_coords)

    salcs.sort_to("blocks")

    ref = reference_data[fn]

    assert len(salcs) >= len(ref)

    for s, expected in enumerate(ref):

        np.testing.assert_allclose(
            salcs[s].coeffs,
            expected["coeffs"],
            atol=1e-4,
        )

 
        result = molsym.salcs.salc_tools.maps_to_negative(
            symtext,
            salcs[s],
            tol=1e-8,
        )


        assert result == expected["maps_to_negative"], (
            f"{fn}: SALC index {s} expected "
            f"{expected['maps_to_negative']} but got {result}"
        )

@pytest.mark.parametrize("fn", list(symmetric_partner_reference.keys()))
def test_generate_symmetric_partner(fn):
    mol = molsym.Molecule.from_file(TEST_DIR / "xyz" / f"{fn}.xyz")
    symtext = molsym.Symtext.from_molecule(mol)

    cart_coords = molsym.salcs.CartesianCoordinates(symtext)
    salcs = molsym.salcs.ProjectionOp(symtext, cart_coords)
    salcs.sort_to("blocks")

    for ref in symmetric_partner_reference[fn]:
        salc = salcs[ref["salc_index"]]

        np.testing.assert_allclose(
            salc.coeffs,
            ref["salc_coeffs"],
            atol=1e-4,
        )

        pos_dipole, _ = molsym.salcs.salc_tools.generate_symmetric_partner(
            symtext,
            salc,
            ref["neg_dipole"],
            data_type="dipole",
        )

        np.testing.assert_allclose(
            pos_dipole,
            ref["pos_dipole"],
            atol=1e-5,
        )

        pos_gradient, _ = molsym.salcs.salc_tools.generate_symmetric_partner(
            symtext,
            salc,
            ref["neg_gradient"],
            data_type="gradient",
        )

        np.testing.assert_allclose(
            pos_gradient,
            ref["pos_gradient"],
            atol=1e-5,
        )
def test_maps_to_negative_molecule_tolerance():

    mol = molsym.Molecule.from_file(TEST_DIR / "xyz" / "ammonia.xyz")

    symtext = molsym.Symtext.from_molecule(mol)

    cart_coords = molsym.salcs.CartesianCoordinates(symtext)

    salcs = molsym.salcs.ProjectionOp(symtext, cart_coords)

    salcs.sort_to("blocks")

    # This SALC does not map under strict tolerance,
    # but does under the molecule tolerance.

    result = molsym.salcs.salc_tools.maps_to_negative(
        symtext,
        salcs[4],
    )

    assert result is True

MOLECULES = ["water", "ammonia", "methane"]
REFERENCE_ROTATION_ANALYSIS = {
    "water": {
        "reduction": "A_2 + B_1 + B_2",
        "functions": {
            "A_2": ["Rz"],
            "B_1": ["Ry"],
            "B_2": ["Rx"],
        },
    },

    "ammonia": {
        "reduction": "A_2 + E",
        "functions": {
            "A_2": ["Rz"],
            "E": ["Rx", "Ry"],
        },
    },

    "methane": {
        "reduction": "T_1",
        "functions": {
            "T_1": ["Rx", "Ry", "Rz"],
        },
    },
}

REFERENCE_DEGREE2_PRETTY = {
    "water": {
        "A_1": [
            "x²",
            "y²",
            "z²",
        ],
        "A_2": [
            "xy",
        ],
        "B_1": [
            "xz",
        ],
        "B_2": [
            "yz",
        ],
    },

    "ammonia": {
        "A_1": [
            "0.707107x² + 0.707107y²",
            "z²",
        ],
        "E": [
            "0.707107x² - 0.707107y²",
            "-xy",
            "xz",
            "yz",
        ],
    },

    "methane": {
        "A_1": [
            "0.57735x² + 0.57735y² + 0.57735z²",
        ],

        "E": [
            "0.408248x² + 0.408248y² - 0.816497z²",
            "-0.707107x² + 0.707107y²",
        ],

        "T_2": [
            "yz",
            "xz",
            "xy",
        ],
    },
}

def load_symtext(molecule):
    xyz = TEST_DIR / "xyz" / f"{molecule}.xyz"
    mol = molsym.Molecule.from_file(str(xyz))
    return molsym.Symtext.from_molecule(mol)


def test_axial_matrix_proper_rotation():
    A = np.eye(3)
    D = axial_matrix(A, tol=global_tol)
    assert np.allclose(D, np.eye(3))


def test_axial_matrix_improper_operation():
    A = np.diag([1.0, 1.0, -1.0])
    D = axial_matrix(A, tol=global_tol)
    assert np.allclose(D, -A)


def test_character_by_operation():
    mats = np.array([
        np.eye(3),
        -np.eye(3),
    ])

    chars = character_by_operation(mats)

    assert np.allclose(chars, [3.0, -3.0])


def test_monomial_label():
    assert monomial_label((1, 0, 0)) == "x"
    assert monomial_label((0, 1, 0)) == "y"
    assert monomial_label((0, 0, 1)) == "z"
    assert monomial_label((2, 0, 0)) == "x^2"
    assert monomial_label((1, 1, 0)) == "x*y"
    assert monomial_label((0, 0, 0)) == "1"


def test_prettify_polynomial_string():
    raw = "1*x^2 + 2*x*y - 3*z^3"
    pretty = prettify_polynomial_string(raw)

    assert "*" not in pretty
    assert "x²" in pretty
    assert "z³" in pretty

@pytest.mark.parametrize("molecule", MOLECULES)
def test_rotation_analysis_reference(capsys, molecule):
    symtext = load_symtext(molecule)

    rep_mats = [axial_matrix(symel.rrep, global_tol) for symel in symtext.symels]
    op_chars = character_by_operation(rep_mats)
    coeffs = symtext.reduction_coefficients(op_chars, False)

    print("Reduction:")
    print(format_reduction(coeffs, symtext))

    captured = capsys.readouterr().out

    ref = REFERENCE_ROTATION_ANALYSIS[molecule]

    assert "Reduction:" in captured
    assert ref["reduction"] in captured

@pytest.mark.parametrize("molecule", MOLECULES)
def test_axial_vector_pretty_reference(capsys, molecule):
    symtext = load_symtext(molecule)

    fxn_set = AxialVectorFunctions(symtext)
    salcs = ProjectionOp(symtext, fxn_set, project_Eckart=False)
    fxn_set.salc_print_style = "pretty"
    print(salcs)
    captured = capsys.readouterr().out

    ref = REFERENCE_ROTATION_ANALYSIS[molecule]

    for irrep, functions in ref["functions"].items():
        assert f"{irrep}:" in captured

        for fxn in functions:
            assert fxn in captured


@pytest.mark.parametrize("molecule", MOLECULES)
def test_degree2_polynomial_pretty_reference(capsys, molecule):
    symtext = load_symtext(molecule)

    fxn_set = PolynomialFunctions(symtext, degree=2)
    salcs = ProjectionOp(symtext, fxn_set, project_Eckart=False)

    fxn_set.salc_print_style = "pretty"

    print(salcs)
    captured = capsys.readouterr().out

    ref = REFERENCE_DEGREE2_PRETTY[molecule]

    for irrep, functions in ref.items():
        assert f"{irrep}:" in captured

        for fxn in functions:
            assert fxn in captured

INTERNAL_COORDINATE_DEFS = {
    "water": [
        [[0, 1], "R1"],
        [[0, 2], "R2"],
        [[1, 0, 2], "A1"],
    ],
    "ammonia": [
        [[0, 1], "R1"],
        [[0, 2], "R2"],
        [[0, 3], "R3"],
        [[1, 0, 2], "A1"],
        [[2, 0, 3], "A2"],
        [[3, 0, 1], "A3"],
    ],
}

INTERNAL_COORDINATE_PRETTY_REFERENCE = {
    "water": [
        "A_1:",
        "P_00(0): 0.707107R1 + 0.707107R2",
        "P_00(2): A1",
        "B_2:",
        "P_00(0): 0.707107R1 - 0.707107R2",
    ],
    "ammonia": [
        "A_1:",
        "P_00(0): 0.57735R1 + 0.57735R2 + 0.57735R3",
        "P_00(3): 0.57735A1 + 0.57735A2 + 0.57735A3",
        "E:",
        "P_00(0): 0.408248R1 + 0.408248R2 - 0.816497R3",
        "P_10(0): 0.707107R1 - 0.707107R2",
        "P_00(3): 0.816497A1 - 0.408248A2 - 0.408248A3",
        "P_10(3): -0.707107A2 + 0.707107A3",
    ],
}

@pytest.mark.parametrize("molecule", ["water", "ammonia"])
def test_internal_coordinate_symbol_pretty_reference(capsys, molecule):
    symtext = load_symtext(molecule)

    fxn_set = InternalCoordinates(
        symtext,
        INTERNAL_COORDINATE_DEFS[molecule],
    )

    salcs = ProjectionOp(symtext, fxn_set)

    fxn_set.salc_print_style = "pretty"

    print(salcs)
    captured = capsys.readouterr().out

    for expected in INTERNAL_COORDINATE_PRETTY_REFERENCE[molecule]:
        assert expected in captured

