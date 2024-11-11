import numpy as np
from molsym.molecule import *

def rotation_matrix(axis, theta):
    """
    Rotation matrix about an axis by theta in radians.

    :param axis: Cartesian vector defining rotation axis
    :param theta: Angle of rotation in radians
    :type axis: NumPy array of shape (3,)
    :type theta: float
    :return: Matrix defining rotation on column vector
    :rtype: NumPy array of shape (3,3)
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # NOT NORMALIZING AXIS!!!
    M = cos_t * np.eye(3)
    M += sin_t * np.cross(np.eye(3), axis)
    M += (1-cos_t) * np.outer(axis, axis)
    return M

def reflection_matrix(axis):
    """
    Reflection matrix about a plane defined by its normal vector.

    :param axis: Cartesian vector defining the plane normal vector
    :type axis: NumPy array of shape (3,)
    :return: Matrix defining reflection on column vector
    :rtype: NumPy array of shape (3,3)
    """
    M = np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            if i == j:
                M[i,i] = 1 - 2*(axis[i]**2)
            else:
                M[i,j] = -2 * axis[i] * axis[j]
                M[j,i] = M[i,j]
    return M

def inversion_matrix():
    """
    Cartesian inversion matrix.

    :return: Matrix defining inversion
    :rtype: NumPy array of shape(3,3)
    """
    return -1*np.eye(3)

def Cn(axis, n):
    """
    Wrapper around rotation_matrix for producing a C_n rotation about axis.

    :param axis: Cartesian vector defining rotation axis
    :param n: Defines rotation angle by theta = 2 pi / n
    :type axis: NumPy array of shape (3,)
    :type n: int
    :return: Matrix defining proper rotation on column vector
    :rtype: NumPy array of shape (3,3)
    """
    theta = 2*np.pi/n
    return rotation_matrix(axis, theta)

def Sn(axis, n):
    """
    Improper rotation S_n about an axis.
    
    :param axis: Cartesian vector defining rotation axis
    :param n: Defines rotation angle by theta = 2 pi / n
    :type axis: NumPy array of shape (3,)
    :type n: int
    :return: Matrix defining improper rotation on column vector
    :rtype: NumPy array of shape (3,3)
    """
    return np.dot(reflection_matrix(axis), Cn(axis, n))

def isequivalent(A,B):
    """
    Returns True if molecule A and B are equivalent with respect to permutation of like atoms.

    :param A: Molecule A
    :param B: Molecule B
    :type A: molsym.Molecule
    :type B: molsym.Molecule
    :return: True if equivalent, False if not
    :rtype: bool
    """
    if A.tol >= B.tol:
        eq_tol = A.tol
    else:
        eq_tol = B.tol
    matched_already = []
    for i in range(A.natoms):
        for j in range(B.natoms):
            # Reduce search list so large molecules are a bit faster
            if j not in matched_already:
                # Check that masses are equal
                if A.masses[i] == B.masses[j]:
                    # Check if atoms are about at the same Cartesian point
                    zs = abs(A.coords[i,:]-B.coords[j,:])
                    if np.allclose(zs, [0,0,0], atol=eq_tol):
                        matched_already.append(j)
                        break
    # Did we find a match for each atom? If so we win
    if len(matched_already) == A.natoms:
        return True
    return False

def calcmoit(atoms):
    """
    Calculates the moment of inertia tensor for a list of atoms.
    
    :param atoms: Set of atoms
    :type atoms: molsym.Molecule
    :return: Cartesian moment of inertia tensor
    :rtype: NumPy array of shape (3,3)
    """
    I = np.zeros((3,3))
    atoms.translate(atoms.find_com())
    for i in range(3):
        for j in range(3):
            if i == j:
                for k in range(atoms.natoms):
                    I[i,i] += atoms.masses[k]*(atoms.coords[k,(i+1)%3]**2+atoms.coords[k,(i+2)%3]**2)
            else:
                for k in range(atoms.natoms):
                    I[i,j] -= atoms.masses[k]*atoms.coords[k,i]*atoms.coords[k,j]
    return I

def normalize(a):
    """
    Normalize vector a to unit length, return None if the input vector is of zero length.

    :param a: Vector of arbitrary magnitude
    :type a: NumPy array of shape (n,)
    :return: Normalized vector or None if the magnitude of a is less than the global tolerance
    :rtype: NumPy array of shape (n,) or None
    """
    n = np.linalg.norm(a)
    if n <= global_tol:
        return None
    return a / np.linalg.norm(a)

def issame_axis(a, b, tol=global_tol):
    """
    Return True if vectors a and b are colinear within the global tolerance.

    :param a: Vector a
    :param b: Vector b
    :param tol: Tolerance for error
    :type a: NumPy array of shape (n,)
    :type b: NumPy array of shape (n,)
    :type tol: float
    :return: True if vectors are colinear, False if not colinear or if either vector has zero length
    :rtype: bool
    """
    A = normalize(a)
    B = normalize(b)
    if A is None or B is None:
        return False
    d = abs(np.dot(A,B))
    return np.isclose(d, 1.0, atol=tol)

def isfactor(n,a):
    """
    Return True if a divides n.
    
    :type n: int
    :type a: int
    :rtype: bool
    """
    if n % a == 0:
        return True
    else:
        return False

def reduce(n, i):
    """
    Divide n and i by their greatest common divisor g.

    :type n: int
    :type i: int
    :return: Tuple of n/g and i/g
    :rtype: (int, int)
    """
    g = gcd(n, i)
    return n//g, i//g # floor divide to get an int, there should never be a remainder since we are dividing by the gcd

def gcd(A, B):
    """
    A quick implementation of the Euclid algorithm for finding the greatest common divisor between A and B.
    
    :type A: int
    :type B: int
    :return: Greatest common divisor between A and B
    :rtype: int
    """
    a = max(A,B)
    b = min(A,B)
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        r = a % b
        return gcd(b, r)

def divisors(n):
    """
    Returns the divisors of n.
    This isn't meant to handle large numbers, thankfully most point groups have an order less than 100
    
    :type n: int
    :return: List of n's divisors
    :rtype: List[int]
    """
    out = []
    for i in range(n):
        if n % (i+1) == 0:
            out.append(i+1)
    return out

def distance(a,b):
    """
    Euclidean distance between a and b.

    :type a: NumPy array of shape (n,)
    :type b: NumPy array of shape (n,)
    :return: Distance between a and b
    :rtype: float
    """
    return np.sqrt(((a-b)**2).sum())
