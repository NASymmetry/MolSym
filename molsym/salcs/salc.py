import molsym
from molsym import symtools
import numpy as np
import warnings
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
        self.remove_complex = False

    def __getitem__(self, salc_idx):
        return self.salcs[salc_idx]
    
    def __len__(self):
        return len(self.salcs)

    
    def __str__(self) -> str:
        style = getattr(self.fxn_set, "salc_print_style", "default")

        if style == "pretty" and hasattr(self.fxn_set, "print_salcs"):
            return self.fxn_set.print_salcs(self)

        out = ""
        for salc in self.salcs:
            out += str(salc)
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
        if self.symtext.complex and not self.remove_complex:
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

    def _gram_schmidt_partner_block(self, salcs_in_this_irrep, n_pf_sets, d):
        """
        Gram-Schmidt orthogonalize one partner component's copies and apply
        the same (conjugated) transform to the other partner components.

        :type salcs_in_this_irrep: List[int]
        :type n_pf_sets: int
        :type d: int
        """
        B1 = self.basis_transformation_matrix[:, salcs_in_this_irrep[:n_pf_sets]]
        trans_mat = np.eye(n_pf_sets, dtype=B1.dtype)
        for col_idx in range(1, n_pf_sets):
            for gs_idx in range(col_idx):
                proj = np.vdot(B1[:, gs_idx], B1[:, col_idx])
                B1[:, col_idx] -= proj * B1[:, gs_idx]
                trans_mat[:, col_idx] -= proj * trans_mat[:, gs_idx]
            nrm = np.linalg.norm(B1[:, col_idx])
            B1[:, col_idx] /= nrm
            self.salcs[salcs_in_this_irrep[col_idx]].coeffs = B1[:, col_idx]
            trans_mat[:, col_idx] /= nrm
        trans_mat_conj = np.conj(trans_mat)
        for pf_idx in range(1, d):
            pfxn_set = [pf_idx * n_pf_sets + i for i in range(n_pf_sets)]
            Bi = self.basis_transformation_matrix[:, [salcs_in_this_irrep[idx] for idx in pfxn_set]]
            Bi_trans = Bi @ trans_mat_conj
            for Bidx, salc_idx in enumerate(pfxn_set):
                self.salcs[salcs_in_this_irrep[salc_idx]].coeffs = Bi_trans[:, Bidx] / np.linalg.norm(Bi_trans[:, Bidx])

    def _orthogonalize_complex_conjugate_pairs(self):
        """
        Orthogonalize complex-conjugate-paired irreps (e.g. E(1)'/E(2)')
        before remove_complexity splits them into real and imaginary parts.
        """
        handled = set()
        for irrep_idx, irrep in enumerate(self.irreps):
            if irrep_idx in handled or irrep.d != 1 or "(1)" not in irrep.symbol:
                continue
            partner_symbol = irrep.symbol.replace("(1)", "(2)")
            partner_idx = next((i for i, ir in enumerate(self.irreps) if ir.symbol == partner_symbol), None)
            if partner_idx is None:
                continue
            primary_idxs = self.salcs_by_irrep[irrep_idx]
            partner_idxs = self.salcs_by_irrep[partner_idx]
            matched_partner = [next((p for p in partner_idxs if self.ispartner(self.salcs[k], self.salcs[p])), None)
                                for k in primary_idxs]
            if None in matched_partner or len(matched_partner) != len(partner_idxs):
                continue
            self._gram_schmidt_partner_block(primary_idxs + matched_partner, n_pf_sets=len(primary_idxs), d=2)
            handled.add(irrep_idx)
            handled.add(partner_idx)

    def _warn_if_not_orthogonal(self):
        """
        Warn if the un-Gram-Schmidt'd SALCs aren't actually orthogonal.
        """
        for irrep_idx, irrep in enumerate(self.irreps):
            salc_idxs = self.salcs_by_irrep[irrep_idx]
            if len(salc_idxs) < 2:
                continue
            B = self.basis_transformation_matrix[:, salc_idxs]
            gram = np.conj(B).T @ B
            maxoff = np.max(np.abs(gram - np.diag(np.diag(gram))))
            if maxoff > self.tol:
                warnings.warn(
                    f"SALCs for irrep {irrep.symbol} are not orthogonal "
                    f"(max |off-diagonal Gram element| = {maxoff:.4g}); "
                    "call finish_building(orthogonalize=True) if these SALCs "
                    "will be used as a change-of-basis (e.g. transforming a Hessian)."
                )

    def finish_building(self, orthogonalize=False, remove_complexity=False):
        """
            Remove complexities if seperably degenerate.
            If doing Eckart projection, reorthogonalize SALCs.

            :type orthogonalize: bool
            :type remove_complexity: bool
        """
        if remove_complexity and orthogonalize:
            self._orthogonalize_complex_conjugate_pairs()
        if remove_complexity: # TODO: Have symtext for groups with reduced complexity, handling irreps such as E2_1g, E2_2g ---> E2g
            self.remove_complex = True
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
                else:
                    salcs_in_this_irrep = self.salcs_by_irrep[irrep_idx]
                    n_pf_sets = round(len(salcs_in_this_irrep) / irrep.d)
                    self._gram_schmidt_partner_block(salcs_in_this_irrep, n_pf_sets, irrep.d)
        else:
            self._warn_if_not_orthogonal()

    @staticmethod
    def salc_to_phi(salcs, nfxn):
        """
        Stacks SALC coefficient vectors into the columns of a matrix.
    
        :type salcs: molsym.SALC or list of molsym.SALC
        :type nfxn: int
        :rtype: NumPy array of shape (nfxn,len(salcs))
        """
        if not isinstance(salcs, (list, tuple)):
            salcs = [salcs]
    
        Phi = np.zeros((nfxn, len(salcs)))
    
        for col, salc in enumerate(salcs):
            coeffs = np.asarray(salc.coeffs, dtype=float)
    
            if coeffs.shape[0] != nfxn:
                raise ValueError(
                    f"SALC coeff length {coeffs.shape[0]} does not match nfxn={nfxn}"
                )
    
            Phi[:, col] = coeffs
    
        return Phi
    
    
    # Splits a flat, partner-sorted SALC list back into one candidate partner
    # set per irrep copy, for select_dprime_partner_sets to try each in turn.
    @staticmethod
    def grouped_partner_sets(sorted_salcs):
        """
        Groups SALCs (already sorted by partner) into one list per irrep copy.
    
        :type sorted_salcs: list of molsym.SALC
        :rtype: list of list of molsym.SALC
        """
        partner_sets = []
        i = 0
    
        while i < len(sorted_salcs):
            salc = sorted_salcs[i]
            d = salc.irrep.d
            group = sorted_salcs[i:i + d]
    
            if len(group) != d:
                raise ValueError("Incomplete SALC partner set")
    
            if any(s.irrep.symbol != salc.irrep.symbol for s in group):
                raise ValueError("Partner set contains mixed irreps")
    
            partner_sets.append(group)
            i += d
    
        return partner_sets