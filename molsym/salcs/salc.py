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
        return f"SALC from P^{self.irrep}_{self.i}{self.j} ({self.bfxn})\n{self.coeffs}\n"

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

    def sort(self):
        new_salc_list = []
        for irrep in self.irreps:
            for salc in self.salc_list:
                if salc.irrep == irrep:
                    new_salc_list.append(salc)
        self.salc_list = new_salc_list

    def addnewSALC(self, new_salc, irrep_idx):
        #if not np.allclose(new_salc.coeffs, np.zeros(new_salc.coeffs.shape), atol=self.tol):
        check = True
        if self.salc_sets[irrep_idx] is None:
            self.salc_sets[irrep_idx] = symtools.normalize(new_salc.coeffs[None,:])
            self.salc_list.append(new_salc)
            self.sort()
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
                self.sort()

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

    def isequiv(self, salc1, salc2):
        chk1 = salc1.irrep == salc2.irrep
        #chk2 = salc1.sh == salc2.sh
        #chk3 = salc1.atom == salc2.atom
        chk4 = salc1.bfxn == salc2.bfxn
        chk5 = salc1.j == salc2.j
        if chk1 and chk4 and chk5:
            return True
        return False

    @property
    def partner_functions(self):
        # Group partner functions together
        out = [[(0,self.salc_list[0])]]
        groop = [0]
        for sidx, salc in enumerate(self.salc_list[1:]):
            chk = False
            for done_salcs in out:
                if self.isequiv(salc, done_salcs[0][1]):
                    done_salcs.append((sidx+1,salc))
                    groop.append(done_salcs[0][0])
                    chk = True
            if not chk:
                out.append([(sidx+1,salc)])
                groop.append(sidx+1)
        return out, groop

    def project_trans_rot(self):
        np.set_printoptions(suppress=True, precision=8, linewidth=1500)
        if not isinstance(self.fxn_set, CartesianCoordinates):
            raise Exception("Method 'project_trans_rot' only supported for SALCs of Cartesian coordinates")
        Tx, Ty, Tz, Rx, Ry, Rz = self.idk()
        biggun = np.vstack((Tx,Ty,Tz,Rx,Ry,Rz))
        print(biggun)
        print(self.partner_functions)
        mw = []
        for i in range(len(self.symtext.mol)):
            for j in range(3):
                mw.append(self.symtext.mol.masses[i])
        mw = np.sqrt(np.array(mw))
        mw_ss = [ss*mw for ss in self.salc_sets]
        biggistun = np.vstack((biggun, mw * self.basis_transformation_matrix.T))
        for h, irrep in enumerate(self.irreps):
            if self.symtext.chartable.irrep_dims[irrep] > 1:
                # Only orthogonalize first partner functions, then transform partners in same way to preserve alignment
                pass
            else:
                biggerun = np.vstack((biggun, mw_ss[h]))
                q,r = np.linalg.qr(biggerun.T)

    def idk(self):
        mol = self.symtext.mol
        natoms = mol.natoms
        rx, ry, rz = np.zeros(3*natoms), np.zeros(3*natoms), np.zeros(3*natoms)
        x, y, z = np.zeros(3*natoms), np.zeros(3*natoms), np.zeros(3*natoms)
        moit = molsym.molecule.calcmoit(self.symtext.mol)
        evals, evec = np.linalg.eigh(moit)
        for i in range(natoms):
            smass = np.sqrt(mol.masses[i])
            x[3 * i + 0] = smass
            y[3 * i + 1] = smass
            z[3 * i + 2] = smass
            atomx, atomy, atomz = mol.coords[i, 0], mol.coords[i, 1], mol.coords[i, 2]
            tval0 = atomx * evec[0,0] + atomy * evec[1,0] + atomz * evec[2, 0];
            tval1 = atomx * evec[0,1] + atomy * evec[1,1] + atomz * evec[2, 1];
            tval2 = atomx * evec[0,2] + atomy * evec[1,2] + atomz * evec[2, 2];
            rx[3 * i + 0] = (tval1 * evec[0,2] - tval2 * evec[0,1]) * smass
            rx[3 * i + 1] = (tval1 * evec[1,2] - tval2 * evec[1,1]) * smass
            rx[3 * i + 2] = (tval1 * evec[2,2] - tval2 * evec[2,1]) * smass

            ry[3 * i + 0] = (tval2 * evec[0,0] - tval0 * evec[0,2]) * smass
            ry[3 * i + 1] = (tval2 * evec[1,0] - tval0 * evec[1,2]) * smass
            ry[3 * i + 2] = (tval2 * evec[2,0] - tval0 * evec[2,2]) * smass

            rz[3 * i + 0] = (tval0 * evec[0,1] - tval1 * evec[0,0]) * smass
            rz[3 * i + 1] = (tval0 * evec[1,1] - tval1 * evec[1,0]) * smass
            rz[3 * i + 2] = (tval0 * evec[2,1] - tval1 * evec[2,0]) * smass
        return x, y, z, rx, ry, rz