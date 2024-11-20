import numpy as np
from .function_set import FunctionSet

class CartesianCoordinates(FunctionSet):
    """
    FunctionSet for Cartesian coordinates
    """
    def __init__(self, symtext) -> None:
        # xyz on each atom in molecule
        # fxn map is ncart (natom sets of xyz) by nsymel
        fxn_list = [i for i in range(3*len(symtext.mol))]
        super().__init__(symtext, fxn_list)

    def get_fxn_map(self):
        """
        Builds the function map for all of the Cartesian coordinates under each symmetry element.
        
        :rtype: NumPy array of shape (nsymels, 3, 3)
        """
        # Symel by xyz by xyz, maps xyz to xyz under symels
        fxn_map = np.zeros((len(self.symtext), 3, 3))
        #phase_map = None
        for s in range(len(self.symtext)):
            fxn_map[s, :, :] = self.symtext.symels[s].rrep.T
        return fxn_map#, phase_map
 
    def get_symmetry_equiv_functions(self):
        """
        Finds the sets of functions that are invariant under all of the symmetry elements.

        :rtype: List[List[int]]
        """
        symm_equiv = []
        done = []
        xyz = np.array([0,1,2], dtype=int)
        for atom_i in range(len(self.symtext.mol)):
            for xyz_idx in range(3):
                fidx = 3*atom_i + xyz_idx
                if fidx in done:
                    continue
                equiv_set = []
                for sidx in range(len(self.symtext)):
                    newatom = self.symtext.atom_map[atom_i, sidx]
                    notzero = xyz[~np.isclose(self.symtext.symels[sidx].rrep[:,xyz_idx], 0.0, atol=1e-10)]
                    r = [3*newatom+i for i in notzero]
                    equiv_set += r
                reduced_equiv_set = list(set(equiv_set))
                symm_equiv.append(reduced_equiv_set)
                done += reduced_equiv_set
        return symm_equiv

    def special_function(self, salc, coord, sidx, irrmat):
        """
        Defines how to map an internal coordinate under a symmetry operation for the ProjectionOp function.
        
        :type salc: NumPy array of shape (number of internal coordinates,)
        :type coord: int
        :type sidx: int
        :type irrmat: NumPy array of shape (nsymel, irrep.d, irrep.d)
        """
        atom_idx = self.symtext.atom_map[coord//3, sidx]
        cfxn = coord % 3
        xyz = self.fxn_map[sidx,cfxn,:]
        for i in range(3):
            if self.symtext.complex:
                salc[:,:,3*atom_idx+i] += np.conj(irrmat[sidx, :, :]) * xyz[i]
            else:
                salc[:,:,3*atom_idx+i] += irrmat[sidx, :, :] * xyz[i]
        return salc
