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
        return f"SALC from P^{self.irrep}_{self.i}{self.j} ({self.bfxn}) gamma={self.gamma:6.3f}\n{self.coeffs}\n"

class SALCs():
    def __init__(self, symtext, fxn_set) -> None:
        self.tol = symtext.mol.tol
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

    #@property
    #def shape(self, irrep_idx):
    #    if self.salc_sets[irrep_idx] is None:
    #        return (0,0)
    #    else:
    #        return self.salc_sets.shape

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
            # This logic seems to be unnecessary now that there is no outer loop
            if np.linalg.matrix_rank(np.vstack((self.salc_sets[irrep_idx], new_salc.coeffs)), tol=self.tol) <= rank:
                check = False
            if check:
                #S = symtools.normalize(new_salc.coeffs)
                S = new_salc.coeffs
                self.salc_sets[irrep_idx] = np.vstack((self.salc_sets[irrep_idx], S))
                self.salc_list.append(new_salc)

    @property
    def basis_transformation_matrix(self):
        if self.symtext.complex and not self.remove_complexity:
            btm = np.zeros((len(self.fxn_set), len(self)), dtype=np.complex128)
        else:
            btm = np.zeros((len(self.fxn_set), len(self)))
        for idx, salc in enumerate(self.salc_list):
            btm[:,idx] = salc.coeffs
        return btm

    @property
    def sorted_basis_transformation_matrix(self):
        # Columns are SALCs of basis fxns (rows)
        # DOES NOT HAVE SAME ORDER AS salc_list
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
        if self.symtext.complex and ("_1" in salc1.irrep or "_2" in salc1.irrep):
            if "_1" in salc1.irrep:
                if "_2" in salc2.irrep:
                    chk1 = salc1.irrep.replace("_1", "") == salc2.irrep.replace("_2", "")
                else:
                    chk1 = False
            elif "_2" in salc1.irrep:
                if "_1" in salc2.irrep:
                    chk1 = salc1.irrep.replace("_2", "") == salc2.irrep.replace("_1", "")
                else:
                    chk1 = False
            else:
                chk1 = False
        else:
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
        # Natively ordered by projection operator outer index
        out = [[(0,self.salc_list[0])]]
        out2 = [[0]]
        groop = [0]
        for sidx, salc in enumerate(self.salc_list[1:]):
            chk = False
            for idx, done_salcs in enumerate(out):
                if self.ispartner(salc, done_salcs[0][1]):
                    done_salcs.append([sidx+1,salc])
                    out2[idx].append(sidx+1)
                    groop.append(done_salcs[0][0])
                    chk = True
            if not chk:
                out.append([(sidx+1,salc)])
                out2.append([sidx+1])
                groop.append(sidx+1)
        return out, out2

    def finish_building(self, orthogonalize=False, remove_complexity=False):
        """
            List of SALC indices grouped by irreps
            Partner functions
            If doing Eckart projection, reorthogonalize SALCs
        """
        self.salcs_by_irrep = [[] for i in range(len(self.irreps))]
        for irrep_idx, irrep in enumerate(self.irreps):
            for salc_idx, salc in enumerate(self.salc_list):
                if salc.irrep == irrep:
                    self.salcs_by_irrep[irrep_idx].append(salc_idx)

        self.partner_functions, o2 = self.sort_partner_functions()
        self.partner_function_sets_by_irrep = [[] for i in range(len(self.irreps))] # This will have empty lists for complex groups
        for irrep_idx, irrep in enumerate(self.irreps):
            for pf_idx, pf_set in enumerate(o2):
                if self.salc_list[pf_set[0]].irrep == irrep:
                    self.partner_function_sets_by_irrep[irrep_idx].append(pf_set)
        self.remove_complexity = remove_complexity
        if remove_complexity: # TODO: Have symtext for groups with reduced complexity, handling irreps such as E2_1g, E2_2g ---> E2g
            for pf in self.partner_functions:
                if len(pf) == 2:
                    nf = 1/np.sqrt(2.0) # This may affect salc.gamma as well!!!
                    s1_save = self.salc_list[pf[0][0]].coeffs
                    self.salc_list[pf[0][0]].coeffs = nf*(self.salc_list[pf[0][0]].coeffs + self.salc_list[pf[1][0]].coeffs)
                    self.salc_list[pf[1][0]].coeffs = nf*(s1_save - self.salc_list[pf[1][0]].coeffs)/1.0j
            for s in self.salc_list:
                if not np.isclose(np.max(np.abs(np.imag(s.coeffs))), 0):
                    raise Exception("Remove complexity procedure unable to remove imaginary components of SALCs")
                s.coeffs = np.real(s.coeffs)

        if orthogonalize:
            np.set_printoptions(suppress=True, precision=5, linewidth=1500)
            for irrep_idx, irrep in enumerate(self.irreps):
                partner_fxns = self.partner_function_sets_by_irrep[irrep_idx]
                if self.symtext.chartable.irrep_dims[irrep] == 1:
                    B = self.basis_transformation_matrix[:,self.salcs_by_irrep[irrep_idx]]
                    for col in range(1,B.shape[1]):
                        for gs_idx in range(col):
                            proj = np.dot(B[:,gs_idx], B[:,col])
                            B[:,col] -= proj * B[:,gs_idx]
                        B[:,col] /= np.linalg.norm(B[:,col])
                    for idx, salc in enumerate(self.salcs_by_irrep[irrep_idx]):
                        self.salc_list[salc].coeffs = B[:,idx]
                else:
                    first_pfxns = [i[0] for i in partner_fxns]
                    B1 = self.basis_transformation_matrix[:,first_pfxns]
                    # Gram-Schmidt orthogonalize columns of B1
                    trans_mat = np.eye(len(first_pfxns))
                    for col_idx in range(1,len(first_pfxns)):
                        for gs_idx in range(col_idx):
                            proj = np.dot(B1[:,gs_idx],B1[:,col_idx])
                            B1[:,col_idx] -= proj * B1[:,gs_idx]
                            trans_mat[:,col_idx] -= proj * trans_mat[:,gs_idx]
                        nrm = np.linalg.norm(B1[:,col_idx])
                        B1[:,col_idx] /= nrm
                        trans_mat[:,col_idx] /= nrm
                    # Transform other partner function sets according to the Gram-Schmidt orthogonalization of B1
                    for pf_idx in range(self.symtext.chartable.irrep_dims[irrep]):
                        pfxn_set = [i[pf_idx] for i in partner_fxns]
                        Bi = self.basis_transformation_matrix[:,pfxn_set]
                        Bi_trans = Bi @ trans_mat
                        for Bidx,salc_idx in enumerate(pfxn_set):
                            self.salc_list[salc_idx].coeffs = Bi_trans[:,Bidx] / np.linalg.norm(Bi_trans[:,Bidx])
                    

