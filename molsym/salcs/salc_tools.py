import numpy as np
import re
from molsym.molecule import global_tol

def linear_axis(symtext):
    """
    Unit vector along the molecular axis of a linear molecule.

    :type symtext: molsym.Symtext
    :rtype: NumPy array of shape (3,)
    """
    coords = np.asarray(symtext.mol.coords, dtype=float)
    if len(coords) < 2:
        raise ValueError("A molecular axis needs at least two atoms.")
    seps = coords[:, None, :] - coords[None, :, :]
    a, b = np.unravel_index(np.argmax(np.linalg.norm(seps, axis=2)), seps.shape[:2])
    return seps[a, b] / np.linalg.norm(seps[a, b])

def _map_atoms(symtext, R, tol):
    coords = np.asarray(symtext.mol.coords, dtype=float)
    new_coords = coords @ R.T
    atom_map = np.empty(len(coords), dtype=int)
    for a, xyz in enumerate(new_coords):
        for b in range(len(coords)):
            if symtext.mol.atoms[a] == symtext.mol.atoms[b] and np.allclose(xyz, coords[b], atol=tol):
                atom_map[a] = b
                break
        else:
            raise ValueError(f"Atom {a} has no image under the requested operation.")
    return atom_map

def finite_operations(symtext, tol=None):
    """
    Concrete symmetry operations as (Cartesian matrix, atom map) pairs.

    Linear groups hold abstract symmetry elements without matrices, so for them this
    returns the C2v (C_inf_v) or D2h (D_inf_h) subgroup about the molecular axis.
    Every Cartesian SALC that some operation of the linear group sends to its negative
    is also sent to its negative by one of these.

    :type symtext: molsym.Symtext
    :rtype: List[Tuple[NumPy array of shape (3,3), NumPy array of shape (natom,)]]
    """
    if tol is None:
        tol = symtext.mol.tol
    if not symtext.pg.is_linear:
        return [(np.asarray(op.rrep), symtext.atom_map[:, k]) for k, op in enumerate(symtext.symels)]

    axis = linear_axis(symtext)
    perp1 = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    E = np.eye(3)
    reflect = lambda n: E - 2 * np.outer(n, n)
    rotate_c2 = lambda n: 2 * np.outer(n, n) - E
    mats = [E, rotate_c2(axis), reflect(perp1), reflect(perp2)]
    if symtext.pg.family == "D":
        mats += [-E, reflect(axis), rotate_c2(perp1), rotate_c2(perp2)]
    return [(R, _map_atoms(symtext, R, tol)) for R in mats]

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
    for k, (op_mat, op_map) in enumerate(finite_operations(symtext, tol)):
        transformed = np.zeros_like(disp_matrix)
        for a in range(N):
            transformed[op_map[a]] = op_mat @ disp_matrix[a]
        if np.allclose(transformed.flatten(), -salc.coeffs, atol=tol):
            found_op = k
            R = op_mat
            atom_map = op_map
            break

    if found_op is None:
        return None, None

    # Apply the same symmetry to the quantity
    if data_type == "dipole":
        pos_data = R @ neg_data

    elif data_type == "gradient":
        pos_data = np.zeros_like(neg_data)
        for a in range(neg_data.shape[0]):
            pos_data[atom_map[a]] = R @ neg_data[a]

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return pos_data, found_op

def generate_degenerate_partner(symtext, salc, partner_salc, data, data_type="dipole", tol=None):
    """
    Use molecular symmetry to generate a quantity at a displacement along one component of a
    degenerate SALC from the same quantity at the same displacement along another component.

    Implemented for linear groups, where the partner of a Pi (Pi_g, Pi_u) component is that
    component rotated 90 degrees about the molecular axis. Every atom lies on the axis, so the
    rotation maps each atom onto itself.

    Parameters
    ----------
    symtext : MolSym object
        Symmetry context of a linear molecule.
    salc, partner_salc : MolSym SALC objects
        The displaced component and the component to generate data for.
    data : np.ndarray
        The quantity at the displacement(s) along `salc`, Cartesian components last:
            (..., 3) for dipole vectors (e.g. one row per stencil point)
            (N_atoms, 3) for gradients
    data_type : str, optional
        "dipole" or "gradient".

    Returns
    -------
    partner_data : np.ndarray or None
        The quantity at the same displacement(s) along `partner_salc`, or None if no 90 degree
        rotation about the axis takes `salc` to `partner_salc`.
    """
    if not symtext.pg.is_linear:
        raise NotImplementedError("generate_degenerate_partner is only implemented for linear point groups.")
    if data_type not in ("dipole", "gradient"):
        raise ValueError(f"Unsupported data_type: {data_type}")
    if tol is None:
        tol = symtext.mol.tol
    axis = linear_axis(symtext)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rot = np.outer(axis, axis) + cross  # 90 degrees about the axis
    disp = salc.coeffs.reshape(-1, 3)
    for R in (rot, rot.T):
        if np.allclose((disp @ R.T).flatten(), partner_salc.coeffs, atol=tol):
            return np.asarray(data) @ R.T
    return None

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

    for op_mat, op_map in finite_operations(symtext, tol):
        transformed = np.zeros_like(disp_matrix)

        for a in range(N):
            transformed[op_map[a]] = op_mat @ disp_matrix[a]
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
