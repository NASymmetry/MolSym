import numpy as np
from molsym.molecule import *

def rotation_matrix(axis, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # NOT NORMALIZING AXIS!!!
    M = cos_t * np.eye(3)
    M += sin_t * np.cross(np.eye(3), axis)
    M += (1-cos_t) * np.outer(axis, axis)
    return M

def reflection_matrix(axis):
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
    return -1*np.eye(3)

def Cn(axis, n):
    theta = 2*np.pi/n
    return rotation_matrix(axis, theta)

def Sn(axis, n):
    return np.dot(reflection_matrix(axis), Cn(axis, n))

def isequivalent(A,B):
    if A.tol >= B.tol:
        eq_tol = A.tol
    else:
        eq_tol = B.tol
    h = []
    for i in range(A.natoms):
        for j in range(B.natoms):
            if A.masses[i] == B.masses[j]:
                zs = abs(A.coords[i,:]-B.coords[j,:])
                if np.allclose(zs, [0,0,0], atol=eq_tol):
                    h.append(j)
                    break
    if len(h) == A.natoms:
        return True
    return False

def calcmoit(atoms):
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
    n = np.linalg.norm(a)
    if n <= global_tol:
        return None
    return a / np.linalg.norm(a)

def issame_axis(a, b):
    A = normalize(a)
    B = normalize(b)
    if A is None or B is None:
        return False
    d = abs(np.dot(A,B))
    return np.isclose(d, 1.0, atol=global_tol)

def isfactor(n,a):
    if n % a == 0:
        return True
    else:
        return False

def reduce(n, i):
    g = gcd(n, i)
    return n//g, i//g # floor divide to get an int, there should never be a remainder since we are dividing by the gcd

def gcd(A, B):
    # A quick implementation of the Euclid algorithm for finding the greatest common divisor
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
    # This isn't meant to handle large numbers, thankfully most point groups have an order less than 100
    out = []
    for i in range(n):
        if n % (i+1) == 0:
            out.append(i+1)
    return out

def distance(a,b):
    return np.sqrt(((a-b)**2).sum())
