import numpy as np

def generate_symmetric_partner(symtext, salc, neg_data, data_type="dipole", tol=None):
    """
    Use molecular symmetry to generate + displacements from - displacements
    for quantities that transform as vectors (dipoles) or sets of atomic vectors (gradients).

    Parameters
    ----------
    symtext : MolSym object
        Contains symmetry operations and atom mapping information.
    salc : MolSym SALC object
        The symmetry-adapted linear combination of Cartesian coordinates.
    neg_data : np.ndarray
        The quantity at the negative displacement.
        Shape:
            (3,) for dipole vectors
            (N_atoms, 3) for gradients
    data_type : str, optional
        "dipole" or "gradient", controls how the transformation is applied.

    Returns
    -------
    pos_data : np.ndarray
        The symmetry-generated positive displacement quantity.
    found_op : int or None
        Index of the symmetry operation used (if any).
    """

    if tol is None:
        tol = symtext.mol.tol
    N = salc.coeffs.size // 3
    disp_matrix = salc.coeffs.reshape(N, 3)

    # Find symmetry operation R such that R(Q) = -Q
    found_op = None
    R = None
    for k, op in enumerate(symtext.symels):
        transformed = np.zeros_like(disp_matrix)
        for a in range(N):
            b = symtext.atom_map[a, k]
            transformed[b] = op.rrep @ disp_matrix[a]
        if np.allclose(transformed.flatten(), -salc.coeffs, atol=symtext.mol.tol):
            found_op = k
            R = op.rrep
            break

    if found_op is None:
        return None, None

    # Apply the same symmetry to the quantity
    if data_type == "dipole":
        pos_data = R @ neg_data

    elif data_type == "gradient":
        pos_data = np.zeros_like(neg_data)
        for a in range(neg_data.shape[0]):
            b = symtext.atom_map[a, found_op]
            pos_data[b] = R @ neg_data[a]

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return pos_data, found_op
def maps_to_negative(symtext, salc, tol=None):
    """
    Test a SALC for +/- displacement equivalence.

    Parameters
    ----------
    symtext
        A MolSym object that contains the symmetry context for point group of the molecule.

    salc
        (nat * 3) Symmetry-Adapted Linear Combination of Cartesian displacement coordinates.

    tol
        The numerical tolerance used to identify a equivalence between the negative displacement
        and a symmetry operation acting on the displacement.

    Returns
    -------

    bool
        The +/- displacements are equivalent.
    """
    if tol is None:
        tol = symtext.mol.tol
    N = salc.coeffs.size // 3
    disp_matrix = salc.coeffs.reshape(N, 3)
    n_ops = symtext.atom_map.shape[1]

    for k in range(n_ops):
        op = symtext.symels[k]
        transformed = np.zeros_like(disp_matrix)

        for a in range(N):
            b = symtext.atom_map[a,k]
            transformed[b] = op.rrep @ disp_matrix[a]
        transformed_flat = transformed.flatten()

        if np.allclose(transformed_flat, -salc.coeffs, atol=tol):
            return True

    return False
