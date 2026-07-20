import numpy as np
import re
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

    if np.isclose(det, 1.0, atol=tol):
        det = 1.0
    elif np.isclose(det, -1.0, atol=tol):
        det = -1.0
    else:
        raise ValueError(f"Operation determinant is not ±1: det={det}")

    return det * A

def axial_vector_salc_to_string(salc, fxn_set, tol=global_tol):
    terms = []

    for coeff, label in zip(salc.coeffs, fxn_set.labels):
        if abs(coeff) < tol:
            continue

        if np.isclose(coeff, 1.0, atol=tol):
            term = label
        elif np.isclose(coeff, -1.0, atol=tol):
            term = f"-{label}"
        else:
            term = f"{coeff:.6g}*{label}"

        terms.append(term)

    if not terms:
        return "0"

    return " + ".join(terms).replace("+ -", "- ")


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


def internal_coordinate_salc_to_string(salc, fxn_set, tol=global_tol):
    terms = []
    labels = getattr(fxn_set, "labels", None)

    for idx, (coeff, ic) in enumerate(zip(salc.coeffs, fxn_set.ic_list)):
        if abs(coeff) < tol:
            continue

        if labels is not None:
            label = labels[idx]
        else:
            label = getattr(ic, "symbol", None)

            if label is None:
                label = getattr(ic, "label", None)

            if label is None:
                label = str(ic)

        if np.isclose(coeff, 1.0, atol=tol):
            term = label
        elif np.isclose(coeff, -1.0, atol=tol):
            term = f"-{label}"
        else:
            term = f"{coeff:.6g}{label}"

        terms.append(term)

    if not terms:
        return "0"

    return " + ".join(terms).replace("+ -", "- ")


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
        elif np.isclose(coeff, 1.0, atol=tol):
            term = label
        elif np.isclose(coeff, -1.0, atol=tol):
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


class SALCFormatter:
    """
    Printable wrapper for SALCs.

    The SALC objects remain generic numerical containers. The FunctionSet
    decides how coefficient vectors should be rendered.
    """

    def __init__(self, salcs):
        self.salcs = salcs

    def __str__(self):
        lines = []

        for irrep in self.salcs.irreps:
            matching = [
                s for s in self.salcs.salcs
                if s.irrep.symbol == irrep.symbol
            ]

            if not matching:
                continue

            lines.append(f"{irrep.symbol}:")

            for s in matching:
                if hasattr(self.salcs.fxn_set, "salc_to_string"):
                    expr = self.salcs.fxn_set.salc_to_string(s)
                else:
                    expr = np.array2string(
                        s.coeffs,
                        precision=3,
                        suppress_small=True,
                    )

                lines.append(f"  P_{s.i}{s.j}({s.bfxn}): {expr}")

        return "\n".join(lines)

def format_salcs(salcs):
    return SALCFormatter(salcs)
