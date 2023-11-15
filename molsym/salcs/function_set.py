import numpy as np
from abc import ABC, abstractmethod

class FunctionSet(ABC):
    def __init__(self, fxn_list, symtext) -> None:
        # fxn_list: List of coordinate objects. Concrete classes will deal with operations on the coordinates
        self.fxns = fxn_list
        self.symtext = symtext
        self.fxn_map, self.phase_map = self.get_fxn_map()
        self.SE_fxns = self.get_symmetry_equiv_functions()

    def __len__(self):
        return len(self.fxns)

    @abstractmethod
    def get_fxn_map(self):
        pass

    @abstractmethod
    def get_symmetry_equiv_functions(self):
        pass

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