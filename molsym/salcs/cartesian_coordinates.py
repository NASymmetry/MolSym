import numpy as np
from .function_set import FunctionSet

class CartesianCoordinates(FunctionSet):
 
    def __init__(self, symtext) -> None:
        # xyz on each atom in molecule
        # fxn map is ncart (natom sets of xyz) by nsymel
        fxn_list = [i for i in range(3*len(symtext.mol))]
        super().__init__(fxn_list, symtext)

    def get_fxn_map(self):
        # Symel by xyz by xyz, maps xyz to xyz under symels
        fxn_map = np.zeros((len(self.symtext), 3, 3))
        phase_map = None
        for s in range(len(self.symtext)):
            fxn_map[s, :, :] = self.symtext.symels[s].rrep
        return fxn_map, phase_map
 
    def get_symmetry_equiv_functions(self):
        symm_equiv = True
