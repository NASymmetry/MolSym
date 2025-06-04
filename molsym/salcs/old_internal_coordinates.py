from copy import deepcopy
import numpy as np
from .function_set import FunctionSet
"""
@codeCoverageIgnore
"""
class InternalCoordinates(FunctionSet):
    """
    FunctionSet for internal coordinates (interatomic distances, angles, dihedral angles, etc.)
    """
    def __init__(self, symtext, fxn_list) -> None:
        self.ic_list = [i[0] for i in fxn_list]
        self.ic_types = [i[1] for i in fxn_list]
        super().__init__(symtext, fxn_list)

    def operate_on_ic(self, ic_idx, symop):
        """
        Maps an internal coordinate to a new internal coordinate under a symmetry operation.
        The phase can be 1 or -1 depending on the effect of the symmetry element.
        :type ic_idx: int
        :type symop: molsym.Symel
        :return: New internal coordinate index and phase
        :rtype: int, float
        """
        mapped_ic = []
        for atom in self.ic_list[ic_idx]:
            atom2 = int(self.symtext.atom_map[atom, symop])
            mapped_ic.append(atom2)
        index, phase = self.ic_index(mapped_ic, symop, ic_idx)
        return index, phase
    
    def get_fxn_map(self):
        """
        Builds the function map for all of the internal coordinates under each symmetry element.

        :rtype: NumPy array of shape (number of internal coordinates, nsymels)
        """
        ic_map = np.zeros((len(self.ic_list), len(self.symtext)), dtype=np.int32)
        S = (len(self.ic_list), len(self.symtext))
        self.phase_map = np.ones(S)
        for ic_idx in range(len(self)):
            for sidx, symel in enumerate(self.symtext.symels):
                index, phase = self.operate_on_ic(ic_idx, sidx)
                ic_map[ic_idx, sidx] = index
                self.phase_map[ic_idx, sidx] *= phase
        return ic_map

    def get_symmetry_equiv_functions(self):
        """
        Finds the sets of functions that are invariant under all of the symmetry elements.

        :rtype: List[List[int]]
        """
        SEICs = []
        done = []
        for ic_idx in range(len(self.ic_list)):
            if ic_idx in done:
                continue
            else:
                seics = []
                for symel_idx in range(len(self.symtext)):
                    seics.append(self.fxn_map[ic_idx, symel_idx])
                reduced_seics = list(set(seics))
                done += reduced_seics
                SEICs.append(reduced_seics)
        
        return SEICs

    def ic_index(self, ic, symop, ic_idx):
        """
        Permutes internal coordinate indices and tracks phase to avoid redundantly defined coordinates.
        Example, the angle between atoms 1,2,3 is the same as 3,2,1.

        :type ic: List[int]
        :rtype: (int, int)
        """
        symbol = self.symtext.symels[symop].symbol
        ic_type = self.ic_types[ic_idx][0]
        phase_from_op = 1.0
        if symbol[0] == "i" or symbol[0] == "S" or symbol[0] == "s": # s is for sigma
            if ic_type in ["D", "O", "L"]:
                phase_from_op = -1.0
        if len(ic) > 3:
            ic2 = deepcopy(ic)
            ic2[2], ic2[3] = ic[3], ic[2]
            for c, coord in enumerate(self.ic_list):
                if ic == coord:
                    return c, 1 * phase_from_op
                elif list(reversed(ic)) == coord:
                    return c, 1 * phase_from_op
                elif ic2 == coord:
                    return c, -1 * phase_from_op
                elif list(reversed(ic2)) == coord:
                    return c, -1 * phase_from_op
        
        elif len(ic) == 3:
            #first, loop through IC list, only after the list is exhausted, loop through reverse indices
            for c, coord in enumerate(self.ic_list):
                if ic == coord:
                    return c, 1
                elif list(reversed(ic)) == coord:
                    return c, 1
            ic2 = deepcopy(ic)
        
        else:
            for c, coord in enumerate(self.ic_list):
                if ic == coord:
                    return c, 1
                elif list(reversed(ic)) == coord:
                    return c, 1

    def span(self, SE_fxn):
        chorker_loaf = np.zeros(self.fxn_map.shape)
        for i in range(self.fxn_map.shape[0]):
            for j in range(self.fxn_map.shape[1]):
                if self.fxn_map[i,j] == i:
                    chorker_loaf[i,j] = 1
        rhorker_loaf = chorker_loaf[SE_fxn, :]
        shorker_loaf = np.sum(rhorker_loaf, axis=0)
        rshorker_loaf = np.zeros(self.symtext.chartable.characters.shape[1])
        for i in range(len(self.symtext)):
            rshorker_loaf[self.symtext.class_map[i]] = shorker_loaf[i]
        span = np.zeros(len(self.symtext.chartable.irreps), dtype=np.int32)
        for idx, irrep in enumerate(self.symtext.chartable.irreps):
            n = round(np.sum(rshorker_loaf * self.symtext.chartable.class_orders * self.symtext.chartable.characters[idx,:]) / self.symtext.order)
            span[idx] = n
        return span
    
    def special_function(self, salc, coord, sidx, irrmat):
        """
        Defines how to map an internal coordinate under a symmetry operation for the ProjectionOp function.
        
        :type salc: NumPy array of shape (number of internal coordinates,)
        :type coord: int
        :type sidx: int
        :type irrmat: NumPy array of shape (nsymel, irrep.d, irrep.d)
        """
        ic2 = self.fxn_map[coord, sidx]
        p = self.phase_map[coord, sidx]
        salc[:,:,ic2] += (irrmat[sidx, :, :]) * p
        return salc
