import molsym
from molsym import symtools
import numpy as np
from dataclasses import dataclass
from .cartesian_coordinates import CartesianCoordinates

# TODO: Need to group SALCs better. Preferably a data structure that allows indexing SALCs by irrep and partner fxn

@dataclass
class SALC():
    coeffs:np.array
    irrep:str
    bfxn:int
    i:int # Outer index of proj. operator, Pij = |Salc_i><Salc_j|
    j:int
    gamma:float # Overlap coefficient of bfxn with SALC. <Salc_j|bfxn>

    def __str__(self) -> str:
        return f"SALC from P^{self.irrep}_{self.i}{self.j} ({self.bfxn}) gamma={self.gamma}\n{self.coeffs}\n"

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

    def __str__(self) -> str:
        out = ""
        for i in self.salc_list:
            out += str(i)
        return out
    
    def __repr__(self) -> str:
        return self.__str__()

    @property
    def shape(self, irrep_idx):
        if self.salc_sets[irrep_idx] is None:
            return (0,0)
        else:
            return self.salc_sets.shape

    def addnewSALC(self, new_salc, irrep_idx):
        check = True
        if self.salc_sets[irrep_idx] is None:
            #self.salc_sets[irrep_idx] = symtools.normalize(new_salc.coeffs[None,:])
            self.salc_sets[irrep_idx] = new_salc.coeffs[None,:]
            self.salc_list.append(new_salc)
        else:
            if self.salc_sets[irrep_idx].shape[0] == 1:
                rank = 1
            else:
                rank = np.linalg.matrix_rank(self.salc_sets[irrep_idx], tol=self.tol)
            if np.linalg.matrix_rank(np.vstack((self.salc_sets[irrep_idx], new_salc.coeffs)), tol=self.tol) <= rank:
                check = False
            if check:
                #S = symtools.normalize(new_salc.coeffs)
                S = new_salc.coeffs
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

    def ispartner(self, salc1, salc2):
        chk1 = salc1.irrep == salc2.irrep
        #chk2 = salc1.sh == salc2.sh
        #chk3 = salc1.atom == salc2.atom
        chk4 = salc1.bfxn == salc2.bfxn
        chk5 = salc1.j == salc2.j
        if chk1 and chk4 and chk5:
            return True
        return False

    def sort_partner_functions(self):
        # Group partner functions together
        out = [[(0,self.salc_list[0])]]
        groop = [0]
        for sidx, salc in enumerate(self.salc_list[1:]):
            chk = False
            for done_salcs in out:
                if self.ispartner(salc, done_salcs[0][1]):
                    done_salcs.append((sidx+1,salc))
                    groop.append(done_salcs[0][0])
                    chk = True
            if not chk:
                out.append([(sidx+1,salc)])
                groop.append(sidx+1)
        return out, groop

    def finish_building(self):
        """
            List of SALC indices grouped by irreps
            Partner functions
        """
        self.salcs_by_irrep = [[] for i in range(len(self.irreps))]
        for irrep_idx, irrep in enumerate(self.irreps):
            for salc_idx, salc in enumerate(self.salc_list):
                if salc.irrep == irrep:
                    self.salcs_by_irrep[irrep_idx].append(salc_idx)
        self.partner_functions = self.sort_partner_functions()

