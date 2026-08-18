import numpy as np

from molsym.molecule import global_tol
from molsym.salcs.polynomial_functions import (
    monomial_exponents,
    polynomial_transformation_matrix,
)

# Residual check for a candidate D': how close T'(g) Phi' is to Phi' D'(g)
# for every operation. Used to reject a bad partner-set candidate.
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

# Entry point for the pseudoinverse/intertwiner method: for each irrep,
# escalate polynomial degree until a partner set is found whose rotated
# SALCs (see rotate_salcs) intertwine correctly with the nonstandard
# frame's own operations (see compute_dprime), then keep that D'.
def select_dprime_partner_sets(standard_symtext,nonstandard_symtext,Q,max_degree,polynomial_function_cls,projection_op_cls,residual_tol=1e-8):
    """
    Selects one good polynomial partner set per irrep and solves for D'.

    :type standard_symtext: molsym.Symtext
    :type nonstandard_symtext: molsym.Symtext
    :type Q: NumPy array of shape (3,3)
    :type max_degree: int
    :type polynomial_function_cls: type, e.g. molsym.salcs.PolynomialFunctions
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
    # solving -- also avoids a real bug where complex-conjugate pairs fail the solve below.
    for irrep in standard_symtext.irreps:
        if irrep.d == 1:
            selected_dprime_mats[irrep.symbol] = list(standard_symtext.irrep_mats[irrep.symbol])

    for degree in range(1, max_degree + 1):
        poly_fxns = polynomial_function_cls(standard_symtext, degree=degree)
        salc_container = projection_op_cls(standard_symtext, poly_fxns)

        salc_container.sort_to("partners")
        partner_sets = grouped_partner_sets(salc_container.salcs)

        Tprime_ops = original_frame_polynomial_ops(nonstandard_symtext.symels,degree)

        for partner_set_idx, salc_set in enumerate(partner_sets):
            irrep = salc_set[0].irrep
            irrep_symbol = irrep.symbol

            if irrep_symbol in selected_dprime_mats:
                continue

            Phi_std = salc_to_phi(salc_set,nfxn=len(poly_fxns.exponents))

            if np.linalg.matrix_rank(Phi_std, tol=1e-10) != irrep.d:
                continue

            Phi_prime = rotate_salcs(Phi_std, Q, degree)

            Dprime_ops, residuals = compute_dprime(Tprime_ops,Phi_prime)

            max_residual = residuals.max()

            if max_residual > residual_tol:
                continue

            selected_dprime_mats[irrep_symbol] = Dprime_ops
            selected_info[irrep_symbol] = {
                "degree": degree,
                "partner_set_idx": partner_set_idx,
                "irrep": irrep,
                "max_residual": max_residual,
                "residuals": residuals,
                "Phi_std": Phi_std,
                "Phi_prime": Phi_prime,
            }

            print(
                f"Selected D' for {irrep_symbol}: "
                f"degree={degree}, partner_set={partner_set_idx}, "
                f"max residual={max_residual:.3e}"
            )

        all_symbols = {ir.symbol for ir in salc_container.irreps}
        if all(symbol in selected_dprime_mats for symbol in all_symbols):
            break

    return selected_dprime_mats, selected_info

# Shared building block: expands a 3x3 Cartesian transformation into its
# induced degree-n representation on the monomial basis.
def poly_rep(A, degree, tol=global_tol):
    """
    Builds the degree-n polynomial representation matrix of A.

    :type A: NumPy array of shape (3,3)
    :type degree: int
    :type tol: float
    :rtype: NumPy array of shape (N,N)
    """
    basis = monomial_exponents(degree)
    return polynomial_transformation_matrix(np.asarray(A, dtype=float), basis)


# Carries a standard-frame SALC over to the nonstandard frame by an exact
# change of coordinates -- no fitting, just re-expressing the same SALC.
def rotate_salcs(Phi_std, Q, degree, tol=global_tol):
    """
    Rotates standard-frame SALC coefficients into the original frame.

    Convention used here:
        x_orig = Q @ x_std

    Therefore:
        p_orig(x_orig) = p_std(Q.T @ x_orig)

    so:
        Phi_orig = P(Q.T) @ Phi_std

    :type Phi_std: NumPy array of shape (nfxn,d)
    :type Q: NumPy array of shape (3,3)
    :type degree: int
    :type tol: float
    :rtype: NumPy array of shape (nfxn,d)
    """
    return poly_rep(Q.T, degree, tol=tol) @ Phi_std


# The nonstandard frame's own T'(g): what select_dprime_partner_sets'
# rotated SALCs actually have to intertwine with.
def original_frame_polynomial_ops(symels, degree, tol=global_tol):
    """
    Builds polynomial T'(g) matrices directly from symel.rrep.

    This assumes symel.rrep has already been rotated into the
    original molecular frame.

    :type symels: list of molsym.Symel
    :type degree: int
    :type tol: float
    :rtype: list of NumPy arrays of shape (N,N)
    """
    Tprime_ops = []

    for symel in symels:
        T = poly_rep(np.asarray(symel.rrep, dtype=float).T, degree, tol=tol)
        Tprime_ops.append(T)

    return Tprime_ops


# The actual solve: D'(g) via pseudoinverse, given a rotated SALC and the
# nonstandard frame's operations. This is the concluding equation.
def compute_dprime(Tprime_ops, Phi_prime, rcond=1e-12):
    """
    Solves D'(g) = Phi_prime^+ T'(g) Phi_prime for every operation.

    :type Tprime_ops: list of NumPy arrays of shape (nfxn,nfxn)
    :type Phi_prime: NumPy array of shape (nfxn,d)
    :type rcond: float
    :return: Dprime_ops is a list of D'_g matrices; residuals is the
        intertwining residual for each operation.
    :rtype: tuple(list, NumPy array)
    """
    Phi_prime = np.asarray(Phi_prime, dtype=float)
    Phi_pinv = np.linalg.pinv(Phi_prime, rcond=rcond)

    Dprime_ops = []
    residuals = []

    for T in Tprime_ops:
        D = Phi_pinv @ T @ Phi_prime
        Dprime_ops.append(D)
        residuals.append(np.linalg.norm(T @ Phi_prime - Phi_prime @ D))

    return Dprime_ops, np.array(residuals)

# Small format-conversion helper: SALC objects -> a plain coefficient matrix
# for the linear algebra in rotate_salcs/compute_dprime.
def salc_to_phi(salcs, nfxn):
    """
    Stacks SALC coefficient vectors into the columns of a matrix.

    :type salcs: molsym.SALC or list of molsym.SALC
    :type nfxn: int
    :rtype: NumPy array of shape (nfxn,len(salcs))
    """
    if not isinstance(salcs, (list, tuple)):
        salcs = [salcs]

    Phi = np.zeros((nfxn, len(salcs)))

    for col, salc in enumerate(salcs):
        coeffs = np.asarray(salc.coeffs, dtype=float)

        if coeffs.shape[0] != nfxn:
            raise ValueError(
                f"SALC coeff length {coeffs.shape[0]} does not match nfxn={nfxn}"
            )

        Phi[:, col] = coeffs

    return Phi


# Splits a flat, partner-sorted SALC list back into one candidate partner
# set per irrep copy, for select_dprime_partner_sets to try each in turn.
def grouped_partner_sets(sorted_salcs):
    """
    Groups SALCs (already sorted by partner) into one list per irrep copy.

    :type sorted_salcs: list of molsym.SALC
    :rtype: list of list of molsym.SALC
    """
    partner_sets = []
    i = 0

    while i < len(sorted_salcs):
        salc = sorted_salcs[i]
        d = salc.irrep.d
        group = sorted_salcs[i:i + d]

        if len(group) != d:
            raise ValueError("Incomplete SALC partner set")

        if any(s.irrep.symbol != salc.irrep.symbol for s in group):
            raise ValueError("Partner set contains mixed irreps")

        partner_sets.append(group)
        i += d

    return partner_sets


# Public entry point called by Symtext.nonstandard_symtext to populate
# nonstandard_symtext.irrep_mats.
def derive_nonstandard_irrep_mats(standard_symtext, nonstandard_symtext, max_degree=10):
    """
    Derives nonstandard-frame irrep matrices by rotating known-correct
    standard-frame SALCs into the new frame and solving for D'(g) directly
    (see select_dprime_partner_sets), rather than deriving them from
    scratch as rotate_irrmats does.

    :type standard_symtext: molsym.Symtext
    :type nonstandard_symtext: molsym.Symtext
    :type max_degree: int
    :rtype: dict mapping irrep_symbol to a NumPy array of shape (nsymel,d,d)
    """
    # deferred import to avoid a circular import with symtext.py -> here -> projection_op -> molsym
    from molsym.salcs.polynomial_functions import PolynomialFunctions
    from molsym.salcs.projection_op import ProjectionOp

    Q = standard_symtext.reverse_rotate
    selected_dprime_mats, _ = select_dprime_partner_sets(
        standard_symtext,
        nonstandard_symtext,
        Q,
        max_degree,
        polynomial_function_cls=PolynomialFunctions,
        projection_op_cls=ProjectionOp,
    )
    missing = {ir.symbol for ir in standard_symtext.irreps} - set(selected_dprime_mats)
    if missing:
        raise RuntimeError(
            f"Could not resolve nonstandard irrep matrices for {sorted(missing)} "
            f"within max_degree={max_degree}"
        )
    return {sym: np.array(mats) for sym, mats in selected_dprime_mats.items()}
