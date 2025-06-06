import numpy as np
import molsym
#from .SymmetryEquivalentIC import *
from .salc import SALC, SALCs
from .cartesian_coordinates import CartesianCoordinates

def project_out_Eckart(eckart_conditions, new_vector):
    """
    Projects out Eckart conditions from the given vector.

    :type eckart_conditions: NumPy array of shape (m,n)
    :type new_vector: NumPy array of shape (n,)
    :rtype: NumPy array of shape (n,)
    """
    for i in range(eckart_conditions.shape[0]):
        new_vector -= np.dot(eckart_conditions[i,:], new_vector) * eckart_conditions[i,:]
    return new_vector

def eckart_conditions(symtext, translational=True, rotational=True):
    """
    Produces a matrix of the Eckart conditions.

    :type symtext: molsym.Symtext
    :rtype: NumPy array of shape (m,n)
    """
    # TODO Needs some cleaning up
    mol = symtext.mol
    natoms = mol.natoms
    rx, ry, rz = np.zeros(3*natoms), np.zeros(3*natoms), np.zeros(3*natoms)
    x, y, z = np.zeros(3*natoms), np.zeros(3*natoms), np.zeros(3*natoms)
    moit = molsym.symtools.calcmoit(symtext.mol)
    evals, evec = np.linalg.eigh(moit)
    for i in range(natoms):
        smass = np.sqrt(mol.masses[i])
        x[3 * i + 0] = smass
        y[3 * i + 1] = smass
        z[3 * i + 2] = smass
        atomx, atomy, atomz = mol.coords[i, 0], mol.coords[i, 1], mol.coords[i, 2]
        tval0 = atomx * evec[0,0] + atomy * evec[1,0] + atomz * evec[2, 0];
        tval1 = atomx * evec[0,1] + atomy * evec[1,1] + atomz * evec[2, 1];
        tval2 = atomx * evec[0,2] + atomy * evec[1,2] + atomz * evec[2, 2];
        rx[3 * i + 0] = (tval1 * evec[0,2] - tval2 * evec[0,1]) * smass
        rx[3 * i + 1] = (tval1 * evec[1,2] - tval2 * evec[1,1]) * smass
        rx[3 * i + 2] = (tval1 * evec[2,2] - tval2 * evec[2,1]) * smass

        ry[3 * i + 0] = (tval2 * evec[0,0] - tval0 * evec[0,2]) * smass
        ry[3 * i + 1] = (tval2 * evec[1,0] - tval0 * evec[1,2]) * smass
        ry[3 * i + 2] = (tval2 * evec[2,0] - tval0 * evec[2,2]) * smass

        rz[3 * i + 0] = (tval0 * evec[0,1] - tval1 * evec[0,0]) * smass
        rz[3 * i + 1] = (tval0 * evec[1,1] - tval1 * evec[1,0]) * smass
        rz[3 * i + 2] = (tval0 * evec[2,1] - tval1 * evec[2,0]) * smass
    t = np.vstack((x,y,z))
    t /= np.linalg.norm(t, axis=1)[:,None]
    r = np.vstack((rx,ry,rz))
    r /= np.linalg.norm(r, axis=1)[:,None]
    both = np.vstack((t,r))
    if not np.isclose(both @ both.T,np.eye(6)).all():
        raise Exception("Eckart conditions not orthogonal")
    if translational and rotational:
        return np.vstack((t,r))
    elif translational:
        return t
    elif rotational:
        return r
    else:
        raise Exception("Calling this function is rather silly if you don't want either output...")

def ProjectionOp(symtext, fxn_set, project_Eckart=True):
    """
    Projection operator: projects the functions in fxn_set into SALCs.

    :type symtext: molsym.Symtext
    :type fxn_set: molsym.FunctionSet
    :rtype: molsym.SALCs
    """
    numred = len(fxn_set)
    salcs = SALCs(symtext, fxn_set)
    orthogonalize = False
    for ir, irrep in enumerate(symtext.irreps):
        if symtext.pg.is_linear:
            irrmat = None
        else:
            irrmat = symtext.irrep_mats[irrep.symbol]
        for se_fxn_set in fxn_set.SE_fxns:
            equivcoord = min(se_fxn_set)
            salc = np.zeros((irrep.d, irrep.d, numred))
            if symtext.complex:
                salc = np.zeros((irrep.d, irrep.d, numred), dtype=np.complex128)
            for sidx in range(len(symtext)):
                salc = fxn_set.special_function(salc, equivcoord, sidx, irrmat)
            salc *= irrep.d/symtext.order
            
            # Project out Eckart conditions when constructing SALCs of Cartesian displacements
            if isinstance(fxn_set, CartesianCoordinates) and project_Eckart:
                orthogonalize = True
                eckart_cond = eckart_conditions(symtext)
                for i in range(irrep.d):
                    for j in range(irrep.d):
                        if not np.allclose(salc[i,j,:], np.zeros(salc[i,j,:].shape), atol=salcs.tol):
                            salc[i,j,:] = project_out_Eckart(eckart_cond, salc[i,j,:])
            # Convert complex SALCs to real
            if symtext.complex and irrep.d==2:
                nf =  1/np.sqrt(2)
                new_i = nf * salc[:,0,:]+salc[:,1,:]
                new_j = nf * (salc[:,0,:]-salc[:,1,:])/1j
                salc[:,0,:] = new_i
                salc[:,1,:] = new_j
            # Add SALCs to SALC object
            for i in range(irrep.d):
                for j in range(irrep.d):
                    if not np.allclose(salc[i,j,:], np.zeros(salc[i,j,:].shape), atol=salcs.tol):
                        gamma = 1.0/np.linalg.norm(salc[i,j,:])
                        salc[i,j,:] = molsym.symtools.normalize(salc[i,j,:])
                        s = SALC(salc[i,j,:], irrep, equivcoord, i, j, gamma)
                        salcs.addnewSALC(s, ir)
    if symtext.complex:
        remove_complexity = True
    else:
        remove_complexity = False
    # Build convenience SALC data structures
    salcs.finish_building(orthogonalize=orthogonalize, remove_complexity=remove_complexity)
    return salcs
