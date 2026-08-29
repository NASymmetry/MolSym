import numpy as np
from copy import deepcopy

from molsym.salcs.spherical_harmonics import sh_rep, SphericalHarmonicFunctions

def build_nonstandard_symtext(symtext, max_degree=10):
    """
    Returns the original-orientation Symtext, with irrep matrices derived
    for that orientation's own symmetry operations.

    :type symtext: molsym.Symtext
    :type max_degree: int, maximum angular momentum l to search before giving up
    :rtype: molsym.Symtext
    """
    if symtext.pg.is_linear:
        raise ValueError(
            f"nonstandard_symtext does not support linear point groups (got {symtext.pg.str}): "
            "their symmetry elements are abstract (rrep=None) and have no discrete "
            "Cartesian rotation to carry over to the original frame."
        )

    standard_symtext = symtext
    nonstandard_symtext = deepcopy(symtext)

    nonstandard_symtext.mol = symtext.mol.transform(symtext.reverse_rotate)
    Q = symtext.reverse_rotate
    for symel in nonstandard_symtext.symels:
        R_std = np.array(symel.rrep, dtype=float)
        symel.rrep = Q @ R_std @ Q.T
        if symel.vector is not None:
            symel.vector = Q @ symel.vector
    nonstandard_symtext.irrep_mats = derive_nonstandard_irrep_mats(standard_symtext, nonstandard_symtext, max_degree=max_degree)
    nonstandard_symtext.is_nonstandard = True
    nonstandard_symtext.assign_dipole_irrep = nonstandard_symtext.dipole_components_to_irrep()

    return nonstandard_symtext

# How close T'(g) Phi' is to Phi' D'(g) for every operation.
def check_dprime_intertwining(Tprime_ops, Phi_prime, Dprime_ops):
    """
    Checks how well D'(g) intertwines with T'(g) through Phi_prime for every operation.

    :type Tprime_ops: list of NumPy arrays of shape (nfxn,nfxn)
    :type Phi_prime: NumPy array of shape (nfxn,d)
    :type Dprime_ops: list of NumPy arrays of shape (d,d)
    :rtype: NumPy array of shape (nsymel,)
    """
    errors = []

    for Tprime_g, Dprime_g in zip(Tprime_ops, Dprime_ops):
        lhs = Tprime_g @ Phi_prime
        rhs = Phi_prime @ Dprime_g
        errors.append(np.linalg.norm(lhs - rhs))

    return np.array(errors)

# For each irrep, escalate angular momentum l until a partner set is found
# in that l-shell whose rotated SALCs intertwine correctly with the
# nonstandard frame's own operations, then keep that D'.
def select_dprime_partner_sets(standard_symtext, nonstandard_symtext, Q, max_l, sh_function_cls, projection_op_cls, residual_tol=1e-8):
    """
    Selects one good spherical-harmonic partner set per irrep and solves for D'.

    :type standard_symtext: molsym.Symtext
    :type nonstandard_symtext: molsym.Symtext
    :type Q: NumPy array of shape (3,3)
    :type max_l: int
    :type sh_function_cls: type, e.g. molsym.salcs.SphericalHarmonicFunctions
    :type projection_op_cls: callable, e.g. molsym.salcs.ProjectionOp
    :type residual_tol: float
    :return: selected_dprime_mats maps irrep_symbol to a list of D'_g
        matrices; selected_info maps irrep_symbol to diagnostics about
        the partner set that was used.
    :rtype: tuple(dict, dict)
    """
    selected_dprime_mats = {}
    selected_info = {}

    # 1D irreps are frame-invariant, so copy them directly instead of
    # solving -- the general solve below fails for complex-conjugate irrep pairs.
    for irrep in standard_symtext.irreps:
        if irrep.d == 1:
            selected_dprime_mats[irrep.symbol] = list(standard_symtext.irrep_mats[irrep.symbol])

    for l in range(1, max_l + 1):
        sh_fxns = sh_function_cls(standard_symtext, l=l)
        salc_container = projection_op_cls(standard_symtext, sh_fxns)

        salc_container.sort_to("partners")
        partner_sets = salc_container.grouped_partner_sets(salc_container.salcs)

        Tprime_ops = original_frame_sh_ops(nonstandard_symtext.symels, l)

        for partner_set_idx, salc_set in enumerate(partner_sets):
            irrep = salc_set[0].irrep
            irrep_symbol = irrep.symbol

            if irrep_symbol in selected_dprime_mats:
                continue

            Phi_std = salc_container.salc_to_phi(salc_set, nfxn=2 * l + 1)

            if np.linalg.matrix_rank(Phi_std, tol=1e-10) != irrep.d:
                continue

            # ProjectionOp doesn't orthogonalize partner SALCs for this
            # FunctionSet (only for CartesianCoordinates), so QR the raw
            # partner set onto an orthonormal basis of the same invariant
            # subspace -- required for compute_dprime's Phi_prime.T shortcut,
            # which only holds when Phi_prime has orthonormal columns.
            Phi_std, _ = np.linalg.qr(Phi_std)

            # sh_rep(A, l) is a direct representation (sh_rep(A,l) sh_rep(B,l)
            # = sh_rep(AB,l)), matching how symel.rrep is used elsewhere with
            # no transpose/inverse.
            Phi_prime = sh_rep(Q, l) @ Phi_std

            Dprime_ops, residuals = compute_dprime(Tprime_ops, Phi_prime)

            max_residual = residuals.max()

            if max_residual > residual_tol:
                continue

            selected_dprime_mats[irrep_symbol] = Dprime_ops
            selected_info[irrep_symbol] = {
                "l": l,
                "partner_set_idx": partner_set_idx,
                "irrep": irrep,
                "max_residual": max_residual,
                "residuals": residuals,
                "Phi_std": Phi_std,
                "Phi_prime": Phi_prime,
            }

        all_symbols = {ir.symbol for ir in salc_container.irreps}
        if all(symbol in selected_dprime_mats for symbol in all_symbols):
            break

    return selected_dprime_mats, selected_info


# The nonstandard frame's own T'(g): what select_dprime_partner_sets'
# rotated SALCs actually have to intertwine with.
def original_frame_sh_ops(symels, l):
    """
    Builds spherical-harmonic T'(g) matrices directly from symel.rrep.

    This assumes symel.rrep has already been rotated into the
    original molecular frame.

    :type symels: list of molsym.Symel
    :type l: int
    :rtype: list of NumPy arrays of shape (2l+1,2l+1)
    """
    return [sh_rep(np.asarray(symel.rrep, dtype=float), l) for symel in symels]


# D'(g) from an exact orthogonal change of basis. Phi_prime has orthonormal
# columns (Phi_std is QR'd in select_dprime_partner_sets before being carried
# over by the orthogonal sh_rep matrix), so its transpose is a left inverse.
def compute_dprime(Tprime_ops, Phi_prime):
    """
    Solves D'(g) = Phi_prime.T T'(g) Phi_prime for every operation.

    :type Tprime_ops: list of NumPy arrays of shape (nfxn,nfxn)
    :type Phi_prime: NumPy array of shape (nfxn,d)
    :return: Dprime_ops is a list of D'_g matrices; residuals is the
        intertwining residual for each operation.
    :rtype: tuple(list, NumPy array)
    """
    Dprime_ops = []
    residuals = []

    for T in Tprime_ops:
        D = Phi_prime.T @ T @ Phi_prime
        Dprime_ops.append(D)
        residuals.append(np.linalg.norm(T @ Phi_prime - Phi_prime @ D))

    return Dprime_ops, np.array(residuals)


def derive_nonstandard_irrep_mats(standard_symtext, nonstandard_symtext, max_degree=10):
    """
    Derives nonstandard-frame irrep matrices by rotating known-correct
    standard-frame SALCs into the new frame and solving for D'(g) directly
    (see select_dprime_partner_sets), rather than deriving them from
    scratch as rotate_irrmats does.

    :type standard_symtext: molsym.Symtext
    :type nonstandard_symtext: molsym.Symtext
    :param max_degree: maximum angular momentum l to search before giving up
    :type max_degree: int
    :rtype: dict mapping irrep_symbol to a NumPy array of shape (nsymel,d,d)
    """
    # deferred import to avoid a circular import with symtext.py -> here -> projection_op -> molsym
    from molsym.salcs.projection_op import ProjectionOp

    Q = standard_symtext.reverse_rotate
    selected_dprime_mats, _ = select_dprime_partner_sets(
        standard_symtext,
        nonstandard_symtext,
        Q,
        max_degree,
        sh_function_cls=SphericalHarmonicFunctions,
        projection_op_cls=ProjectionOp,
    )
    missing = {ir.symbol for ir in standard_symtext.irreps} - set(selected_dprime_mats)
    if missing:
        raise RuntimeError(
            f"Could not resolve nonstandard irrep matrices for {sorted(missing)} "
            f"within max_l={max_degree}"
        )
    return {sym: np.array(mats) for sym, mats in selected_dprime_mats.items()}
