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
