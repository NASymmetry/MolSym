from copy import deepcopy
import numpy as np
from .function_set import FunctionSet

class InternalCoordinates(FunctionSet):
    def __init__(self, fxn_list, symtext) -> None:
        self.ic_list = [i[0] for i in fxn_list]
        self.ic_types = [i[1] for i in fxn_list]
        super().__init__(fxn_list, symtext)

    def operate_on_ic(self, ic_idx, symop):
        symbol = self.symtext.symels[symop].symbol
        ic_type = self.ic_types[ic_idx][0]
        if symbol[0] == "i" or symbol[0] == "S" or symbol[0] == "s": # s is for sigma
            # Are we doing this twice??? TODO here
            if ic_type in ["D", "O", "L"]: 
                self.phase_map[ic_idx, symop] = -1.0
            #if ic_type == "D" or ic_type == "O": 
            #    self.phase_map[ic_idx, symop] = -1.0
            #elif ic_type == "L":
            #    self.phase_map[ic_idx, symop] = -1.0
        mapped_ic = []
        for atom in self.ic_list[ic_idx]:
            atom2 = int(self.symtext.atom_map[atom, symop])
            mapped_ic.append(atom2)
        index, phase = self.ic_index(mapped_ic) 
        return index, phase 
    
    def get_fxn_map(self):
        ic_map = np.zeros((len(self.ic_list), len(self.symtext)), dtype=np.int32)
        S = (len(self.ic_list), len(self.symtext))
        phase_map = np.ones(S)
        for ic_idx in range(len(self)):
            for sidx, symel in enumerate(self.symtext.symels):
                index, phase = self.operate_on_ic(ic_idx, sidx)
                ic_map[ic_idx, sidx] = index
                phase_map[ic_idx, sidx] *= phase
        return ic_map, phase_map

    def get_symmetry_equiv_functions(self):
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

    def ic_index(self, ic):
        if len(ic) > 3:
            ic2 = deepcopy(ic)
            ic2[2], ic2[3] = ic[3], ic[2]
            for f, funky in enumerate(self.ic_list):
                # TODO here
                if ic == funky:
                    return f, 1
                elif list(reversed(ic)) == funky:
                    return f, 1
                elif ic2 == funky:
                    return f, -1
                elif list(reversed(ic2)) == funky:
                    return f, -1
        else:
            for f, funky in enumerate(self.ic_list):
                if ic == funky:
                    return f, 1
                elif list(reversed(ic)) == funky:
                    return f, 1

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