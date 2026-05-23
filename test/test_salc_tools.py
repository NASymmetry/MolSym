import pytest
import numpy as np
import molsym
from pathlib import Path

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
