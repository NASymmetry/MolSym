import numpy as np
from molsym.symtools import *

class RotationElement():
    """
    Data structure holding rotation axes and orders.
    """
    def __init__(self, axis, order):
        self.axis = axis
        self.order = order
    def __eq__(self, other):
        if isinstance(other, RotationElement):
            return issame_axis(self.axis, other.axis) and self.order == other.order

def intersect(a, b):
    """
    Find intersection of sets of RotationElements.

    :type a: List[RotationElement]
    :type b: List[RotationElement]
    :rtype: List[RotationElement]
    """
    out = []
    for re_a in a:
        for re_b in b:
            if re_a == re_b:
                out.append(re_a)
                break
    return out

def rotation_set_intersection(rotation_set):
    """
    Find intersection of all RotationElement sets in rotation_set

    :param rotation_set: Set of sets of RotationElement
    :type rotation_set: List[List[RotationElement]]
    :return: Intersection of rotation sets
    :rtype: List[RotationElement]
    """
    out = rotation_set[0]
    if len(rotation_set) > 1:
        for i in range(len(rotation_set)):
            out = intersect(out, rotation_set[i])
    return out

def find_rotation_sets(mol, SEAs):
    """
    For each set of symmetry equivalent atoms, find the set of possible RotationElements

    :type mol: molsym.Molecule
    :type SEAs: List[molsym.SEA]
    :rtype: List[List[RotationElement]]
    """
    out_all_SEAs = []
    for sea in SEAs:
        length = len(sea.subset)
        out_per_SEA = []
        if length < 2:
            sea.label = "Single Atom"
        elif length == 2:
            sea.label = "Linear"
            sea.axis = normalize(mol[sea.subset[0]].coords)
        else:
            sea_mol = mol[sea.subset]
            sea_mol.translate(sea_mol.find_com())
            evals, evecs = np.linalg.eigh(calcmoit(mol[sea.subset]))
            idx = evals.argsort()
            Ia, Ib, Ic = evals[idx]
            Iav, Ibv, Icv = [evecs[:,i] for i in idx]
            if np.isclose(Ia, Ib, atol=mol.tol) and np.isclose(Ia, Ic, atol=mol.tol):
                sea.label = "Spherical"
            elif np.isclose(Ia+Ib, Ic, atol=mol.tol):
                axis = Icv
                sea.axis = axis
                if np.isclose(Ia, Ib, atol=mol.tol):
                    sea.label = "Regular Polygon"
                    for i in range(2,length+1):
                        if isfactor(length,i):
                            re = RotationElement(axis,i)
                            out_per_SEA.append(re)
                else:
                    sea.label = "Irregular Polygon"
                    for i in range(2,length):
                        if isfactor(length, i):
                            re = RotationElement(axis, i)
                            out_per_SEA.append(re)
            else:
                if not (np.isclose(Ia, Ib, atol=mol.tol) or np.isclose(Ib, Ic, atol=mol.tol)):
                    sea.label = "Asymmetric Rotor"
                    for i in [Iav, Ibv, Icv]:
                        re = RotationElement(i, 2)
                        out_per_SEA.append(re)
                else:
                    if np.isclose(Ia, Ib, atol=mol.tol):
                        sea.label = "Oblate Symmetric Top"
                        axis = Icv
                        sea.axis = Icv
                    else:
                        sea.label = "Prolate Symmetric Top"
                        axis = Iav
                        sea.axis = Iav
                    k = length//2
                    for i in range(2,k+1):
                        if isfactor(k,i):
                            re = RotationElement(axis, i)
                            out_per_SEA.append(re)
            if len(out_per_SEA) > 0:
                out_all_SEAs.append(out_per_SEA)
    return out_all_SEAs

def find_rotations(mol, rotation_set):
    """
    Find the RotationElements in rotation_set that leave the molecule indistinguishable.

    :type rotation_set: List[List[RotationElement]]
    :rtype: List[RotationElement]
    """
    if len(rotation_set) < 1:
        return []
    molmoit = calcmoit(mol)
    evals = np.sort(np.linalg.eigh(molmoit)[0])
    if evals[0] == 0.0 and np.isclose(evals[1], evals[2], atol=mol.tol):
        for i in range(np.shape(mol.coords)[0]):
            if normalize(mol.coords[i,:]) is not None:
                axis = normalize(mol.coords[0,:])
        re = RotationElement(axis, 0)
        return [re]
    rsi = rotation_set_intersection(rotation_set)
    out = []
    for i in rsi:
        rmat = Cn(i.axis, i.order)
        molB = mol.transform(rmat)
        if isequivalent(mol, molB):
            out.append(i)
    return out

def find_a_c2(mol, SEAs):
    """
    Search for any possible C_2 rotation axes, return the first one found.

    :type mol: molsym.Molecule
    :type SEAs: List[molsym.SEA]
    :rtype: NumPy array of shape (3,)
    """
    for sea in SEAs:
        a = c2a(mol, sea)
        if a is not None:
            return a
        else:
            b = c2b(mol, sea)
            if b is not None:
                return b
            else:
                if sea.label == "Linear":
                    for sea2 in SEAs:
                        if sea == sea2:
                            continue
                        elif sea2.label == "Linear":
                            c = c2c(mol, sea, sea2)
                            if c is not None:
                                return c
    return None

def is_there_ortho_c2(mol, SEAs, paxis):
    """
    Search for any possible C_2 rotation axes that are orthogonal to paxis, return the first one found.

    :type mol: molsym.Molecule
    :type SEAs: List[molsym.SEA]
    :type paxis: NumPy array of shape (3,)
    :return: True if found and new C_2 axis
    :rtype: (bool, NumPy array of shape (3,) or None)
    """
    for sea in SEAs:
        b = c2b(mol, sea, axis=paxis)
        if b is not None:
            return True, b
        else:
            a = c2a(mol, sea, axis=paxis)
            if a is not None:
                return True, a
            else:
                if sea.label == "Linear":
                    for sea2 in SEAs:
                        if sea == sea2:
                            continue
                        elif sea2.label == "Linear":
                            c = c2c(mol, sea, sea2, axis=paxis)
                            if c is not None:
                                return True, c
    return False, None

def num_C2(mol, SEAs):
    """
    Find the number of C_2 axis present and the axes defining them.
    
    :type mol: molsym.Molecule
    :type SEAs: List[molsym.SEA]
    :return: Number of C_2 axes and the C_2 axes
    :rtype: (int, List[NumPy array of shape (3,)])
    """
    axes = []
    for sea in SEAs:
        a = c2a(mol, sea, all=True)
        if a is not None:
            for i in a:
                axes.append(i)
        b = c2b(mol, sea, all=True)
        if b is not None:
            for i in b:
                axes.append(i)
    if len(axes) < 1:
        return None
    unique_axes = [axes[0]]
    for i in axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    return len(unique_axes), unique_axes

def c2a(mol, sea, axis=None, all=False):
    """
    Find C_2 axes by testing vectors formed from the origin and midpoint of all pairs of symmetry equivalent atoms.

    :param axis: If not None, only search for C_2 axes that are not equivalent to axis
    :param all: If False return first C_2 found, if True search all atom pairs for C_2 axes
    :type mol: molsym.Molecule
    :type sea: molsym.SEA
    :type axis: None or NumPy array of shape (3,)
    :type all: bool
    :return: C_2 axis or list of C_2 axes (if all=True)
    :rtype: NumPy array of shape (3,) or List[NumPy array of shape (3,)]
    """
    length = len(sea.subset)
    out = []
    for i in range(length):
        for j in range(i+1,length):
            midpoint = mol.coords[sea.subset[i],:] + mol.coords[sea.subset[j],:]
            if np.isclose(midpoint, [0,0,0], atol=mol.tol).all():
                continue
            else:
                midpoint = normalize(midpoint)
                if axis is not None and issame_axis(midpoint, axis) or midpoint is None:
                    continue
                c2 = Cn(midpoint, 2)
                molB = mol.transform(c2)
                if isequivalent(mol, molB):
                    if all:
                        out.append(midpoint)
                    else:
                        return midpoint
    if len(out) < 1:
        return None
    return out

def c2b(mol, sea, axis=None, all=False):
    """
    Find C_2 axes by testing vectors which pass through individual symmetry equivalent atoms.

    :param axis: If not None, only search for C_2 axes that are not equivalent to axis
    :param all: If False return first C_2 found, if True search all atom pairs for C_2 axes
    :type mol: molsym.Molecule
    :type sea: molsym.SEA
    :type axis: None or NumPy array of shape (3,)
    :type all: bool
    :return: C_2 axis or list of C_2 axes (if all=True)
    :rtype: NumPy array of shape (3,) or List[NumPy array of shape (3,)]
    """
    length = len(sea.subset)
    out = []
    for i in range(length):
        c2_axis = normalize(mol.coords[sea.subset[i],:])
        if c2_axis is None:
            continue
        if axis is not None and issame_axis(c2_axis, axis):
            continue
        c2 = Cn(c2_axis, 2)
        molB = mol.transform(c2)
        if isequivalent(mol, molB):
            if all:
                out.append(c2_axis)
            else:
                return c2_axis
    if len(out) < 1:
        return None
    return out

def c2c(mol, sea1, sea2, axis=None):
    """
    Find C_2 axes by testing vectors mutually orthogonal to sets of linear SEAs.

    :param axis: If not None, only search for C_2 axes that are not equivalent to axis
    :param all: If False return first C_2 found, if True search all atom pairs for C_2 axes
    :type mol: molsym.Molecule
    :type sea: molsym.SEA
    :type axis: None or NumPy array of shape (3,)
    :type all: bool
    :return: C_2 axis or list of C_2 axes (if all=True)
    :rtype: NumPy array of shape (3,) or List[NumPy array of shape (3,)]
    """
    rij = mol.coords[sea1.subset[0],:] - mol.coords[sea1.subset[1],:]
    rkl = mol.coords[sea2.subset[0],:] - mol.coords[sea2.subset[1],:]
    c2_axis = normalize(np.cross(rij, rkl))
    if c2_axis is None:
        return None
    if axis is not None and issame_axis(c2_axis, axis):
        return None
    c2 = Cn(c2_axis,2)
    molB = mol.transform(c2)
    if isequivalent(mol, molB):
        return c2_axis
    return None

def highest_order_axis(rotations): 
    """
    Sorts rotations by highest order rotation axis first.

    :type rotations: List[RotationElement]
    :rtype: List[RotationElemtns]
    """
    ns = []
    for i in range(len(rotations)):
        ns.append(rotations[i].order)
    return np.sort(ns)[-1]

def is_there_sigmah(mol, paxis):
    """
    Check for reflection plane with same normal axis as paxis.

    :type mol: molsym.Molecule
    :type paxis: NumPy array of shape (3,)
    :rtype: bool
    """
    sigmah = reflection_matrix(paxis)
    molB = mol.transform(sigmah)
    return isequivalent(mol, molB)

def is_there_sigmav(mol, SEAs, paxis):
    """
    Check for reflection planes with normal axis orthogonal to paxis.

    :type mol: molsym.Molecule
    :type SEAs: List[molsym.SEA]
    :type paxis: NumPy array of shape (3,)
    :rtype: bool
    """
    axes = []
    for sea in SEAs:
        length = len(sea.subset)
        if length < 2:
            continue
        A = sea.subset[0]
        for i in range(1,length):
            B = sea.subset[i]
            #n = normalize(mol[A].xyz - mol[B].xyz)
            n = normalize(mol.coords[A,:] - mol.coords[B,:])
            if n is not None:
                sigma = reflection_matrix(n)
                molB = mol.transform(sigma)
                if isequivalent(mol, molB):
                    axes.append(n)
    if len(axes) < 1:
        if mol_is_planar(mol):
            return True, planar_mol_axis(mol)
        else:
            return False, None
    unique_axes = [axes[0]]
    for i in axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    for i in unique_axes:
        if issame_axis(i, paxis):
            continue
        else:
            return True, i
    return False, None

def mol_is_planar(mol):
    """
    Check if all atoms in the molecue lie in a plane.

    :type mol: molsym.Molecule
    :rtype: bool
    """
    rank = np.linalg.matrix_rank(mol.coords, tol=mol.tol)
    if rank < 3:
        return True
    return False

def planar_mol_axis(mol):
    """
    Returns the normal axis to the plane of a planar molecule.

    :type mol: molsym.Molecule
    :rtype: NumPy array of shape (3,) or None
    """
    for i in range(mol.natoms):
        for j in range(i,mol.natoms):
            a = normalize(mol.coords[i,:])
            b = normalize(mol.coords[j,:])
            if a is not None and b is not None:
                chk = np.dot(a,b)
                if not np.isclose(chk, 1.0, atol=mol.tol):
                    out = normalize(np.cross(a,b))
                    if out is not None:
                        return out
                    #return normalize(np.cross(a,b))
    #return None

def find_C3s_for_Ih(mol):
    """
    Finds the twenty C3 axes for an Ih point group so the paxis and saxis can be defined.

    :type mol: molsym.Molecule
    :rtype: List[NumPy array of shape (3,)]
    """
    c3_axes = []
    for i in range(mol.natoms):
        for j in range(mol.natoms):
            for k in range(mol.natoms):
                if i != j and i != k:
                    rij = mol.coords[i,:] - mol.coords[j,:]
                    rjk = mol.coords[j,:] - mol.coords[k,:]
                    rik = mol.coords[i,:] - mol.coords[k,:]
                    nij = np.linalg.norm(rij)
                    njk = np.linalg.norm(rjk)
                    nik = np.linalg.norm(rik)
                    if np.isclose(nij, njk, atol=mol.tol) and np.isclose(nij, nik, atol=mol.tol):
                        c3_axis = normalize(np.cross(rij, rjk))
                        if c3_axis is not None:
                            c3 = Cn(c3_axis, 3)
                            molB = mol.transform(c3)
                            if isequivalent(mol, molB):
                                c3_axes.append(c3_axis)
    unique_axes = [c3_axes[0]]
    for i in c3_axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    chk = len(unique_axes)
    if chk != 10:
        raise Exception(f"Unexpected number of C3 axes for Ih point group: Found {chk} unique C3 axes")
    return unique_axes

def find_C4s_for_Oh(mol):
    """
    Finds the three C4 axes for an Oh point group so the paxis and saxis can be defined.
    
    :type mol: molsym.Molecule
    :rtype: List[NumPy array of shape (3,)]
    """
    c4_axes = []
    for i in range(mol.natoms):
        for j in range(mol.natoms):
            for k in range(mol.natoms):
                for l in range(mol.natoms):
                    if i != j and k != l and i != k:
                        rij = mol.coords[i,:] - mol.coords[j,:]
                        rjk = mol.coords[j,:] - mol.coords[k,:]
                        rkl = mol.coords[k,:] - mol.coords[l,:]
                        ril = mol.coords[i,:] - mol.coords[l,:]
                        nij = np.linalg.norm(rij)
                        njk = np.linalg.norm(rjk)
                        nkl = np.linalg.norm(rkl)
                        nil = np.linalg.norm(ril)
                        if np.isclose(nij, njk, atol=mol.tol) and np.isclose(nkl, nil, atol=mol.tol) and np.isclose(nij, nkl, atol=mol.tol):
                            c4_axis = normalize(np.cross(rij, rjk))
                            if c4_axis is not None:
                                c4 = Cn(c4_axis, 4)
                                molB = mol.transform(c4)
                                if isequivalent(mol, molB):
                                    c4_axes.append(c4_axis)
    unique_axes = [c4_axes[0]]
    for i in c4_axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    chk = len(unique_axes)
    if chk != 3:
        raise Exception("Unexpected number of C4 axes for Oh point group: Found $(chk) unique C4 axes")
    return unique_axes