from molsym import symtools
import numpy as np
from dataclasses import dataclass

@dataclass
class SALC():
    coeffs:np.array
    irrep:str
    bfxn:int
    i:int # Outer index of proj. operator, Pij = |Salc_i><Salc_j|
    j:int
    gamma:float # Overlap coefficient of bfxn with SALC. <Salc_j|bfxn>

class SALCs():
    def __init__(self, symtext, fxn_set) -> None:
        self.tol = 1e-12
        self.symtext = symtext
        self.fxn_set = fxn_set
        self.irreps = symtext.chartable.irreps
        self.salc_list = []
        self.salc_sets = [None for i in range(len(self.irreps))]

    def __getitem__(self, salc_idx):
        return self.salc_list[salc_idx]
    
    def __len__(self):
        return len(self.salc_list)

    @property
    def shape(self, irrep_idx):
        if self.salc_sets[irrep_idx] is None:
            return (0,0)
        else:
            return self.salc_sets.shape

    def addnewSALC(self, new_salc, irrep_idx):
        #if not np.allclose(new_salc.coeffs, np.zeros(new_salc.coeffs.shape), atol=self.tol):
        check = True
        if self.salc_sets[irrep_idx] is None:
            self.salc_sets[irrep_idx] = symtools.normalize(new_salc.coeffs[None,:])
            self.salc_list.append(new_salc)
        else:
            if self.salc_sets[irrep_idx].shape[0] == 1:
                rank = 1
            else:
                rank = np.linalg.matrix_rank(self.salc_sets[irrep_idx], tol=self.tol)
            if np.linalg.matrix_rank(np.vstack((self.salc_sets[irrep_idx], new_salc.coeffs)), tol=self.tol) <= rank:
                check = False
            if check:
                S = symtools.normalize(new_salc.coeffs)
                self.salc_sets[irrep_idx] = np.vstack((self.salc_sets[irrep_idx], S))
                self.salc_list.append(new_salc)

    @property
    def basis_transformation_matrix(self):
        # Columns are SALCs of basis fxns (rows)
        ctr = 0
        btm = np.zeros((len(self.fxn_set),len(self)))
        for irrep_idx, irrep in enumerate(self.irreps):
            for pf_idx in range(self.symtext.chartable.irrep_dims[irrep]):
                for salc in self.salc_list:
                    if salc.irrep == irrep and salc.i == pf_idx:
                        btm[:,ctr] = salc.coeffs
                        ctr += 1
        return btm