import numpy as np
from math import comb
from .function_set import DenseFunctionSet
from .salc_tools import format_salcs, polynomial_salc_to_string


class PolynomialFunctions(DenseFunctionSet):
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
        return str(format_salcs(salcs))

    def salc_to_string(self, salc):
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

            T = polynomial_transformation_matrix(A, self.exponents)

            # T[row, col] maps input col -> output row.
            # Store as input_idx, output_idx for special_function convenience.
            fxn_map[sidx, :, :] = T.T

        return fxn_map
    
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


def multinomial_terms_for_power(linear_coeffs, power):
    """
    Expand a linear polynomial raised to an integer power using the
    multinomial theorem.

    Computes:
        (ax*x + ay*y + az*z)^power

    and returns the resulting polynomial as a dictionary mapping
    exponent tuples to coefficients.

    :type linear_coeffs: np.ndarray
    :type power: int
    :return: Polynomial dictionary mapping exponent tuples to coefficients
    :rtype: dict[tuple[int,int,int], float]
    """
    terms = {}
    ax, ay, az = linear_coeffs

    for i in range(power + 1):
        for j in range(power - i + 1):
            k = power - i - j

            terms[(i, j, k)] = (
                comb(power, i)
                * comb(power - i, j)
                * (ax ** i)
                * (ay ** j)
                * (az ** k)
            )

    return terms


def multiply_poly_dicts(p1, p2):
    """
    Multiply two multivariate polynomial dictionaries.

    Polynomial dictionaries map exponent tuples to coefficients:
        (a,b,c) -> coeff

    :type p1: dict
    :type p2: dict
    :return: Product polynomial dictionary
    :rtype: dict
    """
    out = {}

    for e1, c1 in p1.items():
        for e2, c2 in p2.items():
            exp = tuple(e1[i] + e2[i] for i in range(3))
            out[exp] = out.get(exp, 0.0) + c1 * c2

    return out


def transform_monomial(exp, A):
    """
    Transform one Cartesian monomial under a coordinate transformation.

    The coordinate transformation is defined by:
        x' = A[0,0]x + A[0,1]y + A[0,2]z
        y' = A[1,0]x + A[1,1]y + A[1,2]z
        z' = A[2,0]x + A[2,1]y + A[2,2]z

    The transformed monomial is expanded into the Cartesian monomial basis.

    :type exp: tuple[int,int,int]
    :type A: np.ndarray
    :return: Polynomial dictionary representation of transformed monomial
    :rtype: dict[tuple[int,int,int], float]
    """
    a, b, c = exp

    px = multinomial_terms_for_power(A[0, :], a)
    py = multinomial_terms_for_power(A[1, :], b)
    pz = multinomial_terms_for_power(A[2, :], c)

    poly = {(0, 0, 0): 1.0}
    poly = multiply_poly_dicts(poly, px)
    poly = multiply_poly_dicts(poly, py)
    poly = multiply_poly_dicts(poly, pz)

    return poly


def polynomial_transformation_matrix(A, basis):
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

    Returns
    -------
    np.ndarray
        Transformation matrix acting on the polynomial basis.
    """
    index = {exp: i for i, exp in enumerate(basis)}
    T = np.zeros((len(basis), len(basis)))

    for col, exp in enumerate(basis):
        transformed = transform_monomial(exp, A)

        for out_exp, coeff in transformed.items():
            row = index[out_exp]
            T[row, col] += coeff

    return T
