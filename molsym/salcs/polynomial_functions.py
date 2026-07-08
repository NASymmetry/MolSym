import numpy as np
from math import comb
from .function_set import FunctionSet
from molsym.molecule import global_tol


class PolynomialFunctions(FunctionSet):
    """
    FunctionSet for homogeneous Cartesian polynomial monomials.
    Basis functions are exponent tuples:
        (a, b, c) -> x^a y^b z^c
    """

    def __init__(self, symtext, degree=2):
        self.degree = degree
        self.exponents = monomial_exponents(degree)
        fxn_list = list(range(len(self.exponents)))
        super().__init__(symtext, fxn_list)

    def print_salcs(self, salcs):
        from molsym.salcs.salc_tools import format_salcs
        return str(format_salcs(salcs))

    def salc_to_string(self, salc):
        from molsym.salcs.salc_tools import polynomial_salc_to_string
        return polynomial_salc_to_string(salc, self)

    def get_fxn_map(self):
        """
        Build polynomial transformatation matrices for each symmetry operation.

        Shape:
            (nsymel, nfxn, nfxn)

        Convention:
            fxn_map[sidx, input_idx, output_idx] = coefficient
        """
        nfxn = len(self.exponents)
        fxn_map = np.zeros((len(self.symtext), nfxn, nfxn))

        for sidx, symel in enumerate(self.symtext.symels):
            A = np.array(symel.rrep, dtype=float)
            A[np.abs(A) < global_tol] = 0.0

            T = polynomial_transformation_matrix(A,self.exponents,tol=global_tol)

            # T[row, col] maps input col -> output row.
            # Store as input_idx, output_idx for special_function convenience.
            fxn_map[sidx, :, :] = T.T

        return fxn_map
    
    def get_symmetry_equiv_functions(self):
        """
        Construct symmetry-equivalent seed sets for polynomial projection.

        First identifies invariant mixing components under the polynomial
        representation. Then selects enough seed monomials so that the
        union of their group orbits spans the entire component.

        This avoids missing irreducible subspaces in non-cyclic polynomial
        blocks, such as the cubic planar block in C3v:
            {x^3, x^2y, xy^2, y^3}

        where x^3 alone cannot generate the A2 component.

        :return: List of symmetry-equivalent seed sets
        :rtype: list[list[int]]
        """
        nfxn = len(self.exponents)
        done = set()
        seed_sets = []
    
        for start in range(nfxn):
            if start in done:
                continue
    
            component = set([start])
            frontier = [start]
    
            while frontier:
                coord = frontier.pop()
    
                for sidx in range(len(self.symtext)):
                    coeffs = self.fxn_map[sidx, coord, :]
    
                    mapped = [i for i, coeff in enumerate(coeffs) if abs(coeff) > global_tol]
    
                    for idx in mapped:
                        if idx not in component:
                            component.add(idx)
                            frontier.append(idx)
    
            component = sorted(component)
            done.update(component)
    
            component_pos = {idx: pos for pos, idx in enumerate(component)}
    
            current_basis = np.zeros((len(component), 0))
    
            for seed in component:
                orbit_vectors = []
    
                for sidx in range(len(self.symtext)):
                    coeffs = self.fxn_map[sidx, seed, :]
    
                    v = np.zeros(len(component))
    
                    for idx in component:
                        v[component_pos[idx]] = coeffs[idx]
    
                    orbit_vectors.append(v)
    
                orbit_matrix = np.column_stack(orbit_vectors)
    
                if current_basis.shape[1] == 0:
                    old_rank = 0
                    trial = orbit_matrix
                else:
                    old_rank = np.linalg.matrix_rank(current_basis,tol=global_tol)
                    trial = np.column_stack((current_basis, orbit_matrix))
    
                new_rank = np.linalg.matrix_rank(trial,tol=global_tol)
    
                if new_rank > old_rank:
                    # ProjectionOp only uses min(se_fxn_set), so return
                    # this seed as its own set.
                    seed_sets.append([seed])
                    current_basis = trial
    
                if new_rank == len(component):
                    break
    
        return seed_sets

    def special_function(self, salc, coord, sidx, irrmat):
        """
        Apply one symmetry operation to a polynomial monomial basis function
        during projection-operator construction.

        The transformed monomial is expanded in the polynomial basis and
        accumulated into the SALC tensor with the appropriate irrep matrix
        weighting.

        :type salc: np.ndarray
        :type coord: int
        :type sidx: int
        :type irrmat: np.ndarray
        :return: Updated SALC tensor
        :rtype: np.ndarray
        """
        coeffs = self.fxn_map[sidx, coord, :]

        for out_idx, coeff in enumerate(coeffs):
            if abs(coeff) < global_tol:
                continue

            if self.symtext.complex:
                salc[:, :, out_idx] += np.conj(irrmat[sidx, :, :]) * coeff
            else:
                salc[:, :, out_idx] += irrmat[sidx, :, :] * coeff

        return salc

def monomial_exponents(degree):
    """
    Generate exponent tuples for homogeneous Cartesian monomials of
    fixed total degree.

    Example for degree 2:
        [(2,0,0), (1,1,0), (1,0,1), ...]

    corresponding to:
        x^2, xy, xz, ...

    :type degree: int
    :return: List of exponent tuples (a,b,c)
    :rtype: list[tuple[int,int,int]]
    """
    basis = []

    for a in range(degree, -1, -1):
        for b in range(degree - a, -1, -1):
            c = degree - a - b
            basis.append((a, b, c))

    return basis


def multinomial_terms_for_power(linear_coeffs, power, tol=global_tol):
    """
    Expand a linear polynomial raised to an integer power using the
    multinomial theorem.

    Computes:
        (ax*x + ay*y + az*z)^power

    and returns the resulting polynomial as a dictionary mapping
    exponent tuples to coefficients.

    :type linear_coeffs: np.ndarray
    :type power: int
    :type tol: float
    :return: Polynomial dictionary mapping exponent tuples to coefficients
    :rtype: dict[tuple[int,int,int], float]
    """
    terms = {}
    ax, ay, az = linear_coeffs

    for i in range(power + 1):
        for j in range(power - i + 1):
            k = power - i - j

            coeff = (
                comb(power, i)
                * comb(power - i, j)
                * (ax ** i)
                * (ay ** j)
                * (az ** k)
            )

            if abs(coeff) > tol:
                terms[(i, j, k)] = coeff

    return terms


def multiply_poly_dicts(p1, p2, tol=global_tol):
    """
    Multiply two multivariate polynomial dictionaries.

    Polynomial dictionaries map exponent tuples to coefficients:
        (a,b,c) -> coeff

    Terms with coefficients below tolerance are discarded.

    :type p1: dict
    :type p2: dict
    :type tol: float
    :return: Product polynomial dictionary
    :rtype: dict
    """
    out = {}

    for e1, c1 in p1.items():
        for e2, c2 in p2.items():
            exp = tuple(e1[i] + e2[i] for i in range(3))
            out[exp] = out.get(exp, 0.0) + c1 * c2

    return {
        exp: coeff
        for exp, coeff in out.items()
        if abs(coeff) > tol
    }


def transform_monomial(exp, A, tol=global_tol):
    """
    Transform one Cartesian monomial under a coordinate transformation.

    The coordinate transformation is defined by:
        x' = A[0,0]x + A[0,1]y + A[0,2]z
        y' = A[1,0]x + A[1,1]y + A[1,2]z
        z' = A[2,0]x + A[2,1]y + A[2,2]z

    The transformed monomial is expanded into the Cartesian monomial basis.

    :type exp: tuple[int,int,int]
    :type A: np.ndarray
    :type tol: float
    :return: Polynomial dictionary representation of transformed monomial
    :rtype: dict[tuple[int,int,int], float]
    """
    a, b, c = exp

    px = multinomial_terms_for_power(A[0, :], a, tol)
    py = multinomial_terms_for_power(A[1, :], b, tol)
    pz = multinomial_terms_for_power(A[2, :], c, tol)

    poly = {(0, 0, 0): 1.0}
    poly = multiply_poly_dicts(poly, px, tol)
    poly = multiply_poly_dicts(poly, py, tol)
    poly = multiply_poly_dicts(poly, pz, tol)

    return poly


def polynomial_transformation_matrix(A, basis, tol=global_tol):
    """
    Construct the matrix representation of a linear coordinate transformation
    on a homogeneous Cartesian polynomial basis.

    Given a 3×3 matrix ``A`` acting on Cartesian coordinates, this function
    computes the corresponding transformation matrix acting on the polynomial
    basis specified by ``basis``. Each column corresponds to one input basis
    function, and each row contains the coefficients of its transformed
    polynomial expressed in the same basis.

    Parameters
    ----------
    A : np.ndarray
        3×3 linear transformation acting on Cartesian coordinates.
    basis : list[tuple[int, int, int]]
        Homogeneous polynomial basis represented by exponent tuples
        ``(a, b, c)``, corresponding to monomials ``x^a y^b z^c``.
    tol : float
        Values with magnitude below this tolerance are set to zero.

    Returns
    -------
    np.ndarray
        Transformation matrix acting on the polynomial basis.
    """
    index = {exp: i for i, exp in enumerate(basis)}
    T = np.zeros((len(basis), len(basis)))

    for col, exp in enumerate(basis):
        transformed = transform_monomial(exp, A, tol)

        for out_exp, coeff in transformed.items():
            row = index[out_exp]
            T[row, col] += coeff

    T[np.abs(T) < tol] = 0.0
    return T
