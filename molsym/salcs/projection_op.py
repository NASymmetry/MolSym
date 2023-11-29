import numpy as np
import molsym
from .SymmetryEquivalentIC import *
import molsym.symtext.irrep_mats as IrrepMats
from .salc import SALC, SALCs
from .internal_coordinates import InternalCoordinates
from .cartesian_coordinates import CartesianCoordinates

def project_out_Eckart(eckart_conditions, new_vector):
    for i in range(eckart_conditions.shape[0]):
        new_vector -= np.dot(eckart_conditions[i,:], new_vector) * eckart_conditions[i,:]
    return new_vector

def eckart_conditions(symtext, translational=True, rotational=True):
    # TODO Needs some cleaning up
    mol = symtext.mol
    natoms = mol.natoms
    rx, ry, rz = np.zeros(3*natoms), np.zeros(3*natoms), np.zeros(3*natoms)
    x, y, z = np.zeros(3*natoms), np.zeros(3*natoms), np.zeros(3*natoms)
    moit = molsym.molecule.calcmoit(symtext.mol)
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

def ProjectionOp(symtext, fxn_set):
    # TODO The way this function is constructed is stupid. I shouldn't have to use isinstance...
    numred = len(fxn_set)
    salcs = SALCs(symtext, fxn_set)
    for ir, irrep in enumerate(symtext.chartable.irreps):
        irrmat = getattr(IrrepMats, "irrm_" + str(symtext.pg))[irrep]
        dim = np.array(irrmat[0]).shape[0]
        for se_fxn_set in fxn_set.SE_fxns:
            equivcoord = se_fxn_set[0]
            salc = np.zeros((dim, dim, numred))
            for sidx in range(len(symtext)):
                # For now, check type of fxn_set to determine how to build SALCs, eventually this should be handled within the fxn_set somehow
                if isinstance(fxn_set, InternalCoordinates):
                    ic2 = fxn_set.fxn_map[equivcoord, sidx]
                    p = fxn_set.phase_map[equivcoord, sidx]
                    salc[:,:,ic2] += (irrmat[sidx, :, :]) * p
                elif isinstance(fxn_set, CartesianCoordinates):
                    atom_idx = symtext.atom_map[equivcoord//3, sidx]
                    cfxn = equivcoord % 3
                    xyz = fxn_set.fxn_map[sidx,cfxn,:]
                    for i in range(3):
                        salc[:,:,3*atom_idx+i] += irrmat[sidx, :, :] * xyz[i]
            salc *= dim/symtext.order
            if isinstance(fxn_set, CartesianCoordinates):
                # Project out Eckart conditions when constructing SALCs of Cartesian displacements
                eckart_cond = eckart_conditions(symtext)
                for i in range(dim):
                    for j in range(dim):
                        if not np.allclose(salc[i,j,:], np.zeros(salc[i,j,:].shape), atol=1e-12):
                            salc[i,j,:] = project_out_Eckart(eckart_cond, salc[i,j,:])
            for i in range(dim):
                for j in range(dim):
                    if not np.allclose(salc[i,j,:], np.zeros(salc[i,j,:].shape), atol=1e-12):
                        gamma = 1.0/np.linalg.norm(salc[i,j,:])
                        salc[i,j,:] = molsym.symtools.normalize(salc[i,j,:])
                        s = SALC(salc[i,j,:], irrep, equivcoord, i, j, gamma)
                        salcs.addnewSALC(s, ir)
    salcs.finish_building()
    return salcs