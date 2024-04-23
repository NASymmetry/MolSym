import numpy as np
import re
from .internal_coordinates import InternalCoordinates
from .function_set import FunctionSet

class LinearInternalCoordinates(InternalCoordinates):
    def __init__(self, symtext, fxn_list) -> None:
        self.ic_list = [i[0] for i in fxn_list]
        self.ic_types = [i[1] for i in fxn_list]
        self.partners = self.get_LinXY_partners()
        super(InternalCoordinates, self).__init__(symtext, fxn_list)

    def get_fxn_map(self):
        # Different definition, value of n assigned to each coordinate
        ic_map = np.zeros((len(self.ic_list)), dtype=np.int32)
        for ic_idx in range(len(self.ic_list)):
            if "R" in self.ic_types[ic_idx]:
                ic_map[ic_idx] = 0
            elif "LinX" in self.ic_types[ic_idx]:
                ic_map[ic_idx] = 1
            elif "LinY" in self.ic_types[ic_idx]:
                ic_map[ic_idx] = 1
        return ic_map

    def get_symmetry_equiv_functions(self):
        SEICs = []
        done = []
        for ic_idx in range(len(self.ic_list)):
            if ic_idx in done:
                continue
            else:
                seics = []
                for symel_idx in range(len(self.symtext)):
                    #seics.append(self.fxn_map[ic_idx, symel_idx])
                    if self.symtext.symels[symel_idx].symbol in ["S", "C_2'"]:
                        new_coord = [self.symtext.atom_map[i,1] for i in self.ic_list[ic_idx]]
                        ic_result, phase = self.ic_index(new_coord)
                        seics.append(ic_result)
                        if "Lin" in self.ic_types[ic_idx]:
                            seics.append(self.partners[ic_result])
                    else:
                        seics.append(ic_idx)
                        if "Lin" in self.ic_types[ic_idx]:
                            seics.append(self.partners[ic_idx])
                reduced_seics = list(set(seics))
                done += reduced_seics
                SEICs.append(reduced_seics)
        return SEICs

    def get_LinXY_partners(self):
        linXY_partners = []
        for ic_idx in range(len(self.ic_list)):
            if "R" in self.ic_types[ic_idx]:
                linXY_partners.append(ic_idx) # No partner
            elif "LinX" in self.ic_types[ic_idx]:
                linxy_idx = re.search(r"(\d+)", self.ic_types[ic_idx]).groups()[0]
                partner_idx = self.ic_types.index("LinY"+linxy_idx)
                linXY_partners.append(partner_idx)
            elif "LinY" in self.ic_types[ic_idx]:
                linxy_idx = re.search(r"(\d+)", self.ic_types[ic_idx]).groups()[0]
                partner_idx = self.ic_types.index("LinX"+linxy_idx)
                linXY_partners.append(partner_idx)
        return linXY_partners

    def special_function(self, salc, coord, sidx, irrmat):
        n = self.fxn_map[coord]
        if self.partners[coord] == coord:
            # R
            pass
        else:
            # LinXY
            pass
        return salc