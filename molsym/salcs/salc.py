import molsym
from molsym import symtools
import numpy as np
from dataclasses import dataclass
from .cartesian_coordinates import CartesianCoordinates
from molsym.symtext.general_irrep_mats import Irrep

# TODO: Need to group SALCs better. Preferably a data structure that allows indexing SALCs by irrep and partner fxn

@dataclass
class SALC():
    """
    Dataclass for SALC information.
    """
    coeffs:np.array
    irrep:Irrep
    bfxn:int
    i:int # Outer index of proj. operator, Pij = |Salc_i><Salc_j|
    j:int
    gamma:float # Overlap coefficient of bfxn with SALC. <Salc_j|bfxn>

    def __str__(self) -> str:
        return f"SALC from P^{self.irrep.symbol}_{self.i}{self.j} ({self.bfxn}) gamma={self.gamma:6.3f}\n{self.coeffs}\n"

class SALCs():
    """
    Class for building and working with SALCs.
    """
    def __init__(self, symtext, fxn_set) -> None:
        self.tol = symtext.mol.tol
        self.symtext = symtext
        self.fxn_set = fxn_set
        self.irreps = symtext.irreps
        self.salcs = []

    def __getitem__(self, salc_idx):
        return self.salcs[salc_idx]
    
    def __len__(self):
        return len(self.salcs)

    def __str__(self) -> str:
        out = ""
        for i in self.salcs:
            out += str(i)
        return out
    
    def __repr__(self) -> str:
        return self.__str__()

    def addnewSALC(self, new_salc, irrep_idx):
        """
        Adds a new SALC to the SALC list if it is not linearly dependent with previous SALCs.

        :type new_salc: NumPy array of shape (n,)
        :type irrep_idx: int
        """
        sbi = self.salcs_by_irrep[irrep_idx]
        if sbi is None:
            self.salcs.append(new_salc)
        else:
            if len(sbi) == 1:
                rank = 1
            else:
                rank = np.linalg.matrix_rank(self.basis_transformation_matrix[:,sbi].T, tol=self.tol)
            if not np.linalg.matrix_rank(np.vstack((self.basis_transformation_matrix[:,sbi].T, new_salc.coeffs)), tol=self.tol) <= rank:
                # Add new SALC if it increases the rank of the SALC matrix
                self.salcs.append(new_salc)

    @property
    def salcs_by_irrep(self):
        """
        List of SALCs sorted by irreducible representation.

        :rtype: List[List[int]]
        """
        salcs_by_irrep = [[] for i in range(len(self.irreps))]
        for irrep_idx, irrep in enumerate(self.irreps):
            for salc_idx, salc in enumerate(self.salcs):
                if salc.irrep.symbol == irrep.symbol:
                    salcs_by_irrep[irrep_idx].append(salc_idx)
        return salcs_by_irrep

    def sort_to(self, sort_style=None):
        """
        Sort SALCs in place by style
            - 'partners': sort such that partner functions are sequential
            - 'blocks': sort such that transformation yields maximal block diagonalization
            - None: no sort applied, SALCs in native ordering
        """
        if sort_style is None:
            pass
        else:
            perm_list = []
            pfxns = self.sort_partner_functions()
            for irrep in self.irreps:
                if sort_style == 'partners':
                    for pf_set in pfxns:
                        if self.salcs[pf_set[0]].irrep.symbol == irrep.symbol:
                            for pf in pf_set:
                                perm_list.append(pf)
                elif sort_style == 'blocks':
                    for mi in range(irrep.d):
                        for pf_set in pfxns:
                            if self.salcs[pf_set[0]].irrep.symbol == irrep.symbol:
                                for pf in pf_set:
                                    if self.salcs[pf].i % irrep.d == mi:
                                        perm_list.append(pf)
                else:
                    raise Exception(f"Invalid sorting selection: {sort_style}")
            new_salcs = [self.salcs[i] for i in perm_list]
            self.salcs = new_salcs

    @property
    def basis_transformation_matrix(self):
        """
        Function by SALC matrix of coefficients.

        :rtype: NumPy array of shape (n functions, n SALCs)
        """
        if self.symtext.complex and not self.remove_complexity:
            btm = np.zeros((len(self.fxn_set), len(self)), dtype=np.complex128)
        else:
            btm = np.zeros((len(self.fxn_set), len(self)))
        for idx, salc in enumerate(self.salcs):
            btm[:,idx] = salc.coeffs
        return btm

    def ispartner(self, salc1, salc2):
        """
        Determine whether two SALCs are partner functions of each other.

        :type salc1: molsym.SALC
        :type salc2: molsym.SALC
        :rtype: bool
        """
        if self.symtext.complex and ("(1)" in salc1.irrep.symbol or "(2)" in salc1.irrep.symbol):
            if "(1)" in salc1.irrep.symbol:
                if "(2)" in salc2.irrep.symbol:
                    chk1 = salc1.irrep.symbol.replace("(1)", "") == salc2.irrep.symbol.replace("(2)", "")
                else:
                    chk1 = False
            elif "(2)" in salc1.irrep.symbol:
                if "(1)" in salc2.irrep.symbol:
                    chk1 = salc1.irrep.symbol.replace("(2)", "") == salc2.irrep.symbol.replace("(1)", "")
                else:
                    chk1 = False
            else:
                chk1 = False
        else:
            chk1 = salc1.irrep.symbol == salc2.irrep.symbol
        chk2 = salc1.bfxn == salc2.bfxn
        chk3 = salc1.j == salc2.j
        return chk1 and chk2 and chk3

    def sort_partner_functions(self):
        """ 
        Group partner functions together 
        Natively ordered by projection operator outer index
        
        :rtype: List[List[int]]
        """
        out = [[0]]
        for sidx, salc in enumerate(self.salcs[1:]):
            chk = False
            for idx, done_salcs in enumerate(out):
                if self.ispartner(salc, self.salcs[done_salcs[0]]):
                    out[idx].append(sidx+1)
                    chk = True
            if not chk:
                out.append([sidx+1])
        return out

    def finish_building(self, orthogonalize=False, remove_complexity=False):
        """
            Remove complexities if seperably degenerate.
            If doing Eckart projection, reorthogonalize SALCs.

            :type orthogonalize: bool
            :type remove_complexity: bool
        """
        self.remove_complexity = remove_complexity
        if remove_complexity: # TODO: Have symtext for groups with reduced complexity, handling irreps such as E2_1g, E2_2g ---> E2g
            pfxns = self.sort_partner_functions()
            for pf in pfxns:
                if len(pf) == 2:
                    nf = 1/np.sqrt(2.0) # This may affect salc.gamma as well!!!
                    s1_save = self.salcs[pf[0]].coeffs
                    self.salcs[pf[0]].coeffs = nf*(self.salcs[pf[0]].coeffs + self.salcs[pf[1]].coeffs)
                    self.salcs[pf[1]].coeffs = nf*(s1_save - self.salcs[pf[1]].coeffs)/1.0j
            for s in self.salcs:
                if not np.isclose(np.max(np.abs(np.imag(s.coeffs))), 0):
                    raise Exception("Remove complexity procedure unable to remove imaginary components of SALCs")
                s.coeffs = np.real(s.coeffs)
        if orthogonalize:
            np.set_printoptions(suppress=True, precision=5, linewidth=1500)
            self.sort_to("blocks")
            for irrep_idx, irrep in enumerate(self.irreps):
                if irrep.d == 1:
                    B = self.basis_transformation_matrix[:,self.salcs_by_irrep[irrep_idx]]
                    for col in range(1,B.shape[1]):
                        for gs_idx in range(col):
                            proj = np.dot(B[:,gs_idx], B[:,col])
                            B[:,col] -= proj * B[:,gs_idx]
                        B[:,col] /= np.linalg.norm(B[:,col])
                    for idx, salc in enumerate(self.salcs_by_irrep[irrep_idx]):
                        self.salcs[salc].coeffs = B[:,idx]
                    #raise Exception("BEANS")
                else:
                    n_pf_sets = round(len(self.salcs_by_irrep[irrep_idx]) / irrep.d)
                    salcs_in_this_irrep = self.salcs_by_irrep[irrep_idx]
                    B1 = self.basis_transformation_matrix[:,salcs_in_this_irrep]
                    # Gram-Schmidt orthogonalize columns of B1
                    trans_mat = np.eye(n_pf_sets)
                    for col_idx in range(1, n_pf_sets):
                        for gs_idx in range(col_idx):
                            proj = np.dot(B1[:,gs_idx],B1[:,col_idx])
                            B1[:,col_idx] -= proj * B1[:,gs_idx]
                            trans_mat[:,col_idx] -= proj * trans_mat[:,gs_idx]
                        nrm = np.linalg.norm(B1[:,col_idx])
                        B1[:,col_idx] /= nrm
                        self.salcs[salcs_in_this_irrep[col_idx]].coeffs = B1[:,col_idx]
                        trans_mat[:,col_idx] /= nrm
                    B1 = self.basis_transformation_matrix[:,salcs_in_this_irrep]
                    # Transform other partner function sets according to the Gram-Schmidt orthogonalization of B1
                    for pf_idx in range(1,irrep.d):
                        pfxn_set = [pf_idx*n_pf_sets + i for i in range(n_pf_sets)]
                        Bi = self.basis_transformation_matrix[:,[salcs_in_this_irrep[idx] for idx in pfxn_set]]
                        Bi_trans = Bi @ trans_mat
                        for Bidx, salc_idx in enumerate(pfxn_set):
                            self.salcs[salcs_in_this_irrep[salc_idx]].coeffs = Bi_trans[:,Bidx] / np.linalg.norm(Bi_trans[:,Bidx])