import numpy as np
import molsym
from .SymmetryEquivalentIC import *
import molsym.symtext.irrep_mats as IrrepMats
from .salc import SALC, SALCs
from .internal_coordinates import InternalCoordinates
from .cartesian_coordinates import CartesianCoordinates

def ProjectionOp(symtext, fxn_set):
    numred = len(fxn_set)
    #salcs = salc_irreps(symtext.chartable, numred)
    salcs = SALCs(symtext, fxn_set)
    for se_fxn_set in fxn_set.SE_fxns:
        equivcoord = se_fxn_set[0]
        for ir, irrep in enumerate(symtext.chartable.irreps):
            irrmat = getattr(IrrepMats, "irrm_" + str(symtext.pg))[irrep]
            dim = np.array(irrmat[0]).shape[0]
            salc = np.zeros((dim, dim, numred))
            for sidx in range(len(symtext)):
                if isinstance(fxn_set, InternalCoordinates):
                    ic2 = fxn_set.fxn_map[equivcoord, sidx]
                    p = fxn_set.phase_map[equivcoord, sidx]
                    # Needs to loop over all functions a Symel could take the operated function into
                    salc[:,:,ic2] += (irrmat[sidx, :, :]) * p
                elif isinstance(fxn_set, CartesianCoordinates):
                    atom_idx = symtext.atom_map[sidx, equivcoord//3]
                    cfxn = equivcoord % 3
                    xyz = fxn_set.fxn_map[sidx,cfxn,:]
                    salc[:,:,3*atom_idx:3*atom_idx+2] += irrmat[sidx, :, :] * xyz
            salc *= dim/symtext.order
            for i in range(dim):
                for j in range(dim):
                    if not np.allclose(salc[i,j,:], np.zeros(salc[i,j,:].shape), atol=1e-12):
                        gamma = 1.0/np.linalg.norm(salc[i,j,:])
                        s = SALC(salc[i,j,:], irrep, equivcoord, i, j, gamma)
                        salcs.addnewSALC(s, ir)
    return salcs