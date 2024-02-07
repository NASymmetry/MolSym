import pytest
import numpy as np
import molsym

# Internal coordinate SALCs
fns = ["water", "ammonia", "methanol"]#, "methane", "benzene"]
ics_test_set = [([[0,1],"R1"],[[0,2],"R2"],[[1,0,2],"A1"]),
                ([[0,1],"R1"],[[0,2],"R2"],[[0,3],"R3"],[[1,0,2],"A1"],[[2,0,3],"A2"],[[3,0,1],"A3"]),
                ([[0,1],"R1"],[[0,2],"R2"],[[0,3],"R3"],[[0,4],"R4"],[[1,5],"R5"],[[2,0,1],"A1"],[[3,0,1],"A2"],[[4,0,1],"A3"],[[0,1,5],"A4"],[[2,0,1,5],"D1"],[[3,0,1,5],"D2"],[[4,0,1,5],"D3"])
                ]
salcs_test_set = [([0.70710678, 0.70710678, 0], [0, 0, 1], [ 0.70710678, -0.70710678, 0]), 
                  ([ 0.57735027,  0.57735027,  0.57735027,  0.        ,  0.        ,  0.        ],
                   [ 0.        ,  0.        ,  0.        ,  0.57735027,  0.57735027,  0.57735027],
                   [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ,  0.        ],
                   [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ,  0.        ],
                   [ 0.        ,  0.        ,  0.        ,  0.81649658, -0.40824829, -0.40824829],
                   [ 0.        ,  0.        ,  0.        ,  0.        , -0.70710678,  0.70710678]),
                  ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0.70710678, 0.70710678, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0.70710678, 0.70710678, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70710678, 0.70710678],
                   [0, 0, 0.70710678, -0.70710678, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0.70710678, -0.70710678, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70710678, -0.70710678])
                   ]
irrep_labels_test_set = [["A1", "A1", "B2"],
                         ["A1", "A1", "E", "E", "E", "E"],
                         ["A'","A'","A'","A'","A'","A'","A'","A'","A'","A''","A''","A''"]]
@pytest.mark.parametrize("i", [i for i in range(len(fns))])
def test_internal_coordinate_SALCs(i):
    mol = molsym.Molecule.from_file("/home/smg13363/MolSym/test/xyz/"+fns[i]+".xyz")
    mol = molsym.symmetrize(mol)
    symtext = molsym.Symtext.from_molecule(mol)
    ic_fxn_set = molsym.salcs.internal_coordinates.InternalCoordinates(symtext, ics_test_set[i])
    salcs = molsym.salcs.projection_op.ProjectionOp(symtext, ic_fxn_set)
    assert all([np.isclose(salcs_test_set[i][j], salcs.salc_list[j].coeffs).all() for j in range(len(ic_fxn_set))])
    assert [salcs.salc_list[j].irrep for j in range(len(ic_fxn_set))] == irrep_labels_test_set[i]

# Cartesian geometry SALCs with and without Eckart


# Spherical harmonics SALCs TODO

