import numpy as np
from math import comb
import re
from molsym.salcs.projection_op import ProjectionOp
from molsym.salcs.axial_vector_functions import AxialVectorFunctions
from molsym.salcs.polynomial_functions import PolynomialFunctions
from molsym.molecule import global_tol

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

def character_by_operation(rep_mats):
    return np.array([np.trace(D) for D in rep_mats], dtype=float)


def axial_matrix(A, tol=global_tol):
    det = np.linalg.det(A)

    if abs(det - 1.0) < tol:
        det = 1.0
    elif abs(det + 1.0) < tol:
        det = -1.0

    return det * A

def analyze_rotations(symtext, print_pretty=True):
    rep_mats = []

    for i, symel in enumerate(symtext.symels):
        D = axial_matrix(symel.rrep, global_tol)
        rep_mats.append(D)

    op_chars = character_by_operation(rep_mats)
    coeffs = symtext.reduction_coefficients(op_chars, False)

    if print_pretty:
        print("Reduction:")
        print(format_reduction(coeffs, symtext))

    fxn_set = AxialVectorFunctions(symtext)

    salcs = ProjectionOp(symtext,fxn_set,project_Eckart=False)

    if print_pretty:
        print_axial_vector_salcs(salcs)

    return salcs

def axial_vector_salc_to_string(salc, fxn_set, tol=global_tol):
    terms = []

    for coeff, label in zip(salc.coeffs, fxn_set.labels):
        if abs(coeff) < tol:
            continue

        if abs(coeff - 1.0) < tol:
            term = label
        elif abs(coeff + 1.0) < tol:
            term = f"-{label}"
        else:
            term = f"{coeff:.6g}*{label}"

        terms.append(term)

    if not terms:
        return "0"

    return " + ".join(terms).replace("+ -", "- ")


def print_axial_vector_salcs(salcs):
    fxn_set = salcs.fxn_set

    for irrep in salcs.irreps:
        matching = [s for s in salcs.salcs if s.irrep.symbol == irrep.symbol]

        if not matching:
            continue

        print(f"{irrep.symbol}:")

        for s in matching:
            expr = axial_vector_salc_to_string(s, fxn_set)
            print(f"  P_{s.i}{s.j}({s.bfxn}): {expr}")

def monomial_label(exp):
    a, b, c = exp
    pieces = []

    for label, power in zip(("x", "y", "z"), (a, b, c)):
        if power == 0:
            continue
        elif power == 1:
            pieces.append(label)
        else:
            pieces.append(f"{label}^{power}")

    return "*".join(pieces) if pieces else "1"

def format_reduction(coeffs, symtext):
    pieces = []

    for i, mult in enumerate(coeffs):
        if mult == 0:
            continue

        symbol = symtext.irreps[i].symbol

        if mult == 1:
            pieces.append(symbol)
        else:
            pieces.append(f"{mult}{symbol}")

    return " + ".join(pieces) if pieces else "0"


def construct_polynomials(symtext, degree=2, print_pretty=True):
    fxn_set = PolynomialFunctions(symtext, degree=degree)
    salcs = ProjectionOp(symtext, fxn_set, project_Eckart=False)
    if print_pretty:
        print_polynomial_salcs(salcs)
    return salcs

def polynomial_salc_to_string(salc, fxn_set, tol=global_tol, pretty=True):
    """
    Convert one polynomial SALC coefficient vector into a readable polynomial.
    """
    terms = []

    for coeff, exp in zip(salc.coeffs, fxn_set.exponents):
        if abs(coeff) < tol:
            continue

        label = monomial_label(exp)

        if label == "1":
            term = f"{coeff:.6g}"
        elif abs(coeff - 1.0) < tol:
            term = label
        elif abs(coeff + 1.0) < tol:
            term = f"-{label}"
        else:
            term = f"{coeff:.6g}*{label}"

        terms.append(term)

    if not terms:
        return "0"

    out = " + ".join(terms)
    out = out.replace("+ -", "- ")

    if pretty:
        out = prettify_polynomial_string(out)

    return out


def prettify_polynomial_string(poly):
    """
    Minimal string cleanup for character-table-style display.
    """
    superscript_map = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")

    for power in sorted(set(re.findall(r"\^[-+]?\d+", poly)), key=len, reverse=True):
        poly = poly.replace(power, power[1:].translate(superscript_map))

    return poly.replace("*", "")


def print_polynomial_salcs(salcs, pretty=True):
    """
    Print polynomial SALCs grouped by irrep.
    """
    fxn_set = salcs.fxn_set

    for irrep in salcs.irreps:
        matching = [s for s in salcs.salcs if s.irrep.symbol == irrep.symbol]

        if not matching:
            continue

        print(f"{irrep.symbol}:")

        for s in matching:
            poly = polynomial_salc_to_string(s,fxn_set,pretty=pretty)

            print(f"  P_{s.i}{s.j}({s.bfxn}): {poly}")
