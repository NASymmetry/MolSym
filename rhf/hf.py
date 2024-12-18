import psi4
import numpy as np
from input import Settings
import copy
from copy import deepcopy
import molsym
from molsym.molecule import Molecule
from molsym.salcs.RSH_SALCs import Project
from bdmats import BDMatrix
from dpd import DPD
import sys
import time

class SparseERI():
    """
    Object for creating and storing sparse ERI vectors.
    Generates unique indices, reduces indices by symmetry arguments, and grabs integral values from full matrix.
    """
    def __init__(self, nbfxns):
        self.nbfxns = nbfxns
        self.idxs = self.unique_idx()

    #def remove_asym(self, ctab:CharacterTable):
    def whereitgo(self, g, orbital_idxs):
        for i, idx in enumerate(orbital_idxs):
            if g in idx:
                return i
    
    def dp_contains(self, ctab, a, b, c, d):
        chars = a * b * c * d
        h = np.sum(ctab.class_orders)
        s = sum(chars * ctab.class_orders * ctab.characters[0]) #for does it contain TSIR only, index A1 irrep
        n = s / h
        if n > 0:
            return True
        return False

    def get_ints_dummy(self, bigERI):
        v = []
        for i,idx in enumerate(self.idxs):
            v.append(bigERI[idx])
        self.v = v

def get_basis(molecule, basis):
    print("Inside get basis")
    num = molecule.natom()
    molecule_basis = []
    counter = 0
    for x in range(0, num):
        atom_basis = []
        molecule_basis.append(str(molecule.symbol(x)))
        for y in range(0, basis.nshell_on_center(x)):
            atom_basis.append(basis.shell(y+counter).am)
            print(vars(basis.shell(y+counter)))
        counter += basis.nshell_on_center(x)
        molecule_basis.append(atom_basis)
    return molecule_basis

def get_basis_lib(molecule, basis):
    num = molecule.natom()
    molecule_basis = []
    counter = 0
    for x in range(0, num):
        atom_basis = []
        molecule_basis.append(str(molecule.symbol(x)))
        for y in range(0, basis.nshell_on_center(x)):
            atom_basis.append(basis.shell(y+counter).am)
        counter += basis.nshell_on_center(x)
        molecule_basis.append(atom_basis)
    return molecule_basis


def rhf_energy(D, H, F):
    """
    Calculate HF energy
    """
    if isinstance(D, BDMatrix):
        Df = D.full_mat()
        Hf = H.full_mat()
        Ff = F.full_mat()
        E = sum(sum(np.multiply(Df,(Hf+Ff))))
    else:
        E = sum(sum(np.multiply(D,(H+F))))
    return E

def ERI_indicies(irreplength):
    vec = []
    start = 0
    total = 0
    for a, i in enumerate(irreplength):
        if i == 0:
            vec.append(None)
        elif a > 0:
            vec.append([total + 1,total + i])
        else:
            vec.append([start,i])
        total += i
        start = i
    return vec

def eri_2d(ERI):
    twod_eri = np.zeros((ERI.shape[0] * ERI.shape[1], ERI.shape[2] * ERI.shape[3]))
    for i in range(0, ERI.shape[0]):  
        for j in range(0, ERI.shape[1]):  
            for k in range(0, ERI.shape[2]):  
                for l in range(0, ERI.shape[3]):
                    ij = ERI.shape[1] * i + j
                    kl = ERI.shape[3] * k + l
                    twod_eri[ij,kl] = ERI[i,j,k,l]
    return twod_eri
                    
#transforms 2d arrays to 1d using compound index
def twod_oned(mat):
    if len(mat) == 0:
        pass
    else:
        oned_mat = np.zeros((mat.shape[0] * mat.shape[1]))
        for i in range(0, mat.shape[0]):  
            for j in range(0, mat.shape[1]):  
                ij = mat.shape[1] * i + j
                oned_mat[ij] = mat[i,j]
        return oned_mat
#transforms 1d arrays to 2d using compound index
def oned_twod(mat):
    root = int(np.sqrt(mat.shape[0]))
    twod_mat = np.zeros((root, root))
    #lned_mat = np.zeros((mat.shape[0],  mat.shape[1]))
    for i in range(0, twod_mat.shape[0]):  
        for j in range(0, twod_mat.shape[1]):  
            ij = root * i + j
            twod_mat[i,j] = mat[ij]
    return twod_mat

#builds fock using symmettry arguments and broadcasted tensors
def build_fock_blocky_sym(H, Dp, repacked_bigERI, repacked_bigERI_swapped, nbfxns):
    oned_h = [] 
    oned_f = [] 
    oned_d = [] 
    for hi, h in enumerate(H.blocks):
        if len(h) == 0:
            oned_h.append(np.array([]))
            f = np.array([])
            oned_f.append(f)
        else:
            oned_h.append(twod_oned(h))
            f = np.zeros((oned_h[hi].shape))
            oned_f.append(f)
    for d in Dp.blocks:
        if len(d) == 0:
            oned_d.append(np.array([]))
        else:
            oned_d.append(twod_oned(d))
    
    for b, block in enumerate(dpd.nonzero_blocks):
        f_sym, d_sym = block[0], block[3]
        oned_h_s, oned_d_s = oned_h[f_sym], oned_d[d_sym]
        ji = np.einsum('pr,r->p', repacked_bigERI[b], oned_d_s)
        ki = np.einsum('pr,r->p', repacked_bigERI_swapped[b], oned_d_s)
        oned_f[f_sym] += 2 * ji - ki 
    F = []
    for z, hs in enumerate(oned_h):
        oned_f[z] += hs
        if len(hs) == 0:
            F.append(np.array([]))
        else:
            F.append(oned_twod(oned_f[z]))
    return BDMatrix(F)

def build_fock_blocky(H, Dp, ERI, nbfxns):
    print("inside build fock blocky")
    ERI_swapped = np.swapaxes(ERI, 1,2)
    Hfull = H.full_mat()
    Dfull = Dp.full_mat()
    print("creating vectors out of D and H")
    oned_H = twod_oned(Hfull)
    oned_D = twod_oned(Dfull)
    twod_eri = eri_2d(ERI)
    twod_eri_swapped = eri_2d(ERI_swapped)
    Ji = np.einsum('pr,r->p', twod_eri, oned_D)
    Ki = np.einsum('pr,r->p', twod_eri_swapped, oned_D)
    oned_F = oned_H + 2 * Ji - Ki
    print("unpack Fock matrix")
    count = 0
    Fn = oned_twod(oned_F)
    F = []
    for i, I in enumerate(Dp.blocks):
        f = Fn[count:count+len(I),count:count+len(I)]
        count += len(I)
        F.append(f)
    return BDMatrix(F)

def build_fock_chonky(H, Dp, ERI, nbfxns):
    Ji = []
    Ki = []
    Dpfull = Dp.full_mat()
    Ji = np.einsum('pqrs,rs->pq', ERI, Dpfull)
    Ki = np.einsum('prqs,rs->pq', ERI, Dpfull)
    Fn = H.full_mat() + 2*Ji - Ki
    F = []
    count = 0
    for i, I in enumerate(Dp.blocks):
        f = Fn[count:count+len(I),count:count+len(I)]
        count += len(I)
        F.append(f)
    return BDMatrix(F)

#This is the slowest algo for using symmetry in RHF
def build_fock_sparse_sym(H, Dp, sparse_eri, nbfxns):
    Fout = np.zeros(np.shape(H))
    Fout = deepcopy(H)
    Ft = np.zeros((nbfxns, nbfxns))
    D = Dp.full_mat()
    for z, idx in enumerate(sparse_eri.idxs):
        i,j,k,l = idx
        v = sparse_eri.v[z]
        ij = idx2(i,j)
        kl = idx2(k,l)
        yij = i != j
        ykl = k != l
        yab = ij != kl

        Xik = 2.0 if i==k else 1.0
        Xjk = 2.0 if j==k else 1.0
        Xil = 2.0 if i==l else 1.0
        Xjl = 2.0 if j==l else 1.0
        if yij & ykl & yab:
            #J
            Ft[i,j] += 4.0*D[k,l]*v
            Ft[k,l] += 4.0*D[i,j]*v

            # K
            Ft[i,k] -= Xik*D[j,l]*v
            Ft[j,k] -= Xjk*D[i,l]*v
            Ft[i,l] -= Xil*D[j,k]*v
            Ft[j,l] -= Xjl*D[i,k]*v

        elif ykl & yab:
            # J
            Ft[i,j] += 4.0*D[k,l]*v
            Ft[k,l] += 2.0*D[i,j]*v

            # K
            Ft[i,k] -= Xik*D[j,l]*v
            Ft[i,l] -= Xil*D[j,k]*v
        elif yij & yab:
            # J
            Ft[i,j] += 2.0*D[k,l]*v
            Ft[k,l] += 4.0*D[i,j]*v

            # K
            Ft[i,k] -= Xik*D[j,l]*v
            Ft[j,k] -= Xjk*D[i,l]*v

        elif yij & ykl:
            # Only possible if i = k and j = l
            # and i < j ⇒ i < l

            # J
            Ft[i,j] += 4.0*D[k,l]*v

            # K
            Ft[i,k] -= D[j,l]*v
            Ft[i,l] -= D[j,k]*v
            Ft[j,l] -= D[i,k]*v
        elif yab:
            # J
            Ft[i,j] += 2.0*D[k,l]*v
            Ft[k,l] += 2.0*D[i,j]*v
            # K
            Ft[i,k] -= Xik*D[j,l]*v
        else:
            Ft[i,j] += 2.0*D[k,l]*v
            Ft[i,k] -= D[j,l]*v
    offset = 0
    for h, irrep in enumerate(Fout.blocks):
        hlen = np.shape(irrep)[0]
        for i in range(hlen):
            irrep[i,i] += Ft[i+offset,i+offset]
            for j in range(i+1,hlen):
                irrep[i,j] += Ft[i+offset,j+offset] + Ft[j+offset,i+offset]
                irrep[j,i] = irrep[i,j]
        offset += hlen

    return Fout

def build_D(C, eps, ndocc):
    """
    Builds the density matrix. There are two paths depending on whether D is a full matrix or a block diagonal matrix.
    """
    if isinstance(C, BDMatrix):
        T = order_eigval(eps)
        blocks = []
        for h, Cirrep in enumerate(C.blocks):
            Cidxs = []
            for t in T[:ndocc]:
                if t[1] == h:
                    Cidxs.append(t[2])
            irrep_size = np.shape(Cirrep)[0]
            Dblock = np.zeros(np.shape(Cirrep))
            for m in range(irrep_size):
                for n in range(irrep_size):
                    for cidx in Cidxs:
                        Dblock[m,n] += Cirrep[m,cidx]*Cirrep[n,cidx]
            blocks.append(Dblock)
        return BDMatrix(blocks)

    else:
        Cf = C.full_mat()
        Cocc = Cf[:,:ndocc]
        Df = np.einsum("pi,qi->pq", Cocc, Cocc)
        D = []
        sizes = [4,0,2,1]
        ctr = 0
        for i in sizes:
            if i == 0:
                D.append(np.empty((0,)))
            D.append(Df[ctr:ctr+i,ctr:ctr+i])
            ctr += i
        return BDMatrix(D)

def order_eigval(eigval):
    """
    Sort the eigenvalues and eigenvectors from smallest to largest eigenvalue.
    Returns a list of lists containing eigenvalue and indices relating it's postion w.r.t. our irrep conventions.
    """
    B = []
    idx = 0
    for i, irrep in enumerate(eigval):
        for j, e in enumerate(irrep):
            B.append([e,i,j]) # i is the irrep index, j is the orbital index within irrep i, and idx is the index of the eigenvalue w.r.t. the full matrix
            idx += 1
    t = sorted(B, key=lambda x: x[0]) # sort based on eigenvalue
    return t



def aotoso(A, salcs):
    print("looping over irreps")
    """
    AO->SO transformation for one electron integrals
    """
    B = []
    for i, irrep in enumerate(salcs):
        temp1 = np.einsum('uv,ui->iv', A, irrep.lcao, optimize='optimal')
        temp  = np.einsum('iv,vj->ij', temp1, irrep.lcao, optimize='optimal')
        B.append(temp)
    return BDMatrix(B)

def aotoso_2(ERI, salcs):
    """
    AO->SO transformation for two electron integrals
    """
    first = True
    for i, salc in enumerate(salcs):
        if first:
            s = salc.lcao
            first = False
        else:
            if salc.lcao is None:
                pass
            else:
                s = np.concatenate((s,salc.lcao), axis=1)
    temp1 = np.einsum("PQRS,Pp,Qq->pqRS", ERI, s, s, optimize='optimal')
    E = np.einsum("pqRS,Rr,Ss->pqrs", temp1, s, s, optimize='optimal')
    #E = np.einsum("PQRS,Pp,Qq,Rr,Ss->pqrs", ERI, s, s, s, s, optimize='optimal')
    return E

def get_norm(i):
    return 1/np.sqrt(i)

#build antisymmetrizer, normalize overlap so diagonal is 1
def normalize_S(S):
    A = []
    normlists = []
    for i, s in enumerate(S):
        over = np.zeros((len(s),len(s)))
        if len(s) == 0:
            A.append(np.array([]))
            continue
        else:
            normlist = []
            for i in range(len(s)):
                norm1 = get_norm(s[i,i])
                normlist.append(norm1)
                for j in range(len(s)):
                    norm2 = get_norm(s[j,j])
                    over[i,j] = s[i,j] * norm1 * norm2
            eigval, U = np.linalg.eigh(over)
            Us = deepcopy(U)
            for i in range(len(eigval)):
                Us[:,i] = U[:,i] * 1.0/np.sqrt(eigval[i])

            for i in range(len(eigval)):
                Us[i,:] = Us[i,:] * (normlist[i])
            anti = np.dot(Us, U.T)
            A.append(anti)
            normlists.append(normlist)
    return BDMatrix(A)

def rhf_gwh_guess(S, T, V):
    from scipy.linalg import fractional_matrix_power
    #initialize empty f
    F = []
    for I, X in enumerate(S.blocks):
        if len(X) == 0:
            F.append(np.array([]))
        else:
            F.append(np.zeros((len(X),len(X))))
    F = BDMatrix(F)
    
    gwh_k = 1.75
    dummy = 0.0
    H = T + V

    A = []
    Ct = []
    En = []
    #perform gwh guess
    for i, s in enumerate(S.blocks):
        if len(s) == 0:
            #F.append(np.array([]))
            A.append(np.array([]))
            Ct.append(np.array([]))
            En.append(np.array([]))
            continue
        else:
            for fi in range(0, len(s)):
                F.blocks[i][fi, fi] = H.blocks[i][fi, fi]
                for fj in range(0, fi):
                    dummy = 0.5 * gwh_k * (H.blocks[i][fi, fi] + H.blocks[i][fj, fj]) * s[fi, fj]
                    F.blocks[i][fi, fj] = dummy
                    F.blocks[i][fj, fi] = dummy
            #construct A, get intial orbital energies
            A.append(fractional_matrix_power(s, -0.5))
            en_i, ct_i = np.linalg.eigh(F.blocks[i])
            Ct.append(ct_i)
            En.append(en_i)
    print("Initial Orbital GWH Guess")
    print(En)

    A = BDMatrix(A)
    Ct = BDMatrix(Ct)
    Ft = A.dot(F).dot(A)
    C = A.dot(Ct)
    return C, A, En

def rhf_core_guess(S, T, V):
    from scipy.linalg import fractional_matrix_power
    A = []
    F = T + V
    Ct = []
    En = []
    for i, s in enumerate(S.blocks):
        if len(s) == 0:
            A.append(np.array([]))
            En.append(np.array([]))
            Ct.append(np.array([]))
        else:
            A.append(fractional_matrix_power(s, -0.5))
            en_i, ct_i = np.linalg.eigh(F.blocks[i])
            Ct.append(ct_i)
            En.append(en_i)
    print("Initial Orbital Core Guess")
    print(En)
    A = BDMatrix(A)
    Ct = BDMatrix(Ct)
    Ft = A.dot(F).dot(A)
    C = A.dot(Ct)
    return C, A, En


def qc(molecule):
    qc_obj = {
        "symbols": [molecule.symbol(x) for x in range(0, molecule.natom())] ,
        "geometry": molecule.geometry(),
    }
    return qc_obj


#input file and settings

print("do the thing")
molecule = psi4.geometry(Settings['molecule'])
molecule.update_geometry()
ndocc = Settings['nalpha'] #Settings['nbeta']
scf_max_iter = Settings['scf_max_iter']

schema = qc(molecule)

mol = Molecule.from_schema(schema)          
symtext = molsym.Symtext.from_molecule(mol) 
mole = symtext.mol
molecule.set_geometry(psi4.core.Matrix.from_array(symtext.mol.coords))
basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'], puream=True)
Enuc = molecule.nuclear_repulsion_energy()
ints = psi4.core.MintsHelper(basis)

libmsym = False
if libmsym:
    from libmsym_wrap import LibmsymWrapper
    molecule = psi4.geometry(Settings['molecule'])
    molecule.update_geometry()
    molecule_basis = get_basis_lib(molecule, basis)
    exec_libmsym = LibmsymWrapper(molecule, molecule_basis)
    exec_libmsym.run()
    salcs = exec_libmsym.salcs
else:
    salcs, irreplength, reorder_psi4 = molsym.salcs.RSH_SALCs.Project(mole, molecule, basis, symtext)

for s, salc in enumerate(salcs):
    if salc.lcao is None:
        salc.lcao = np.zeros((0,salcs[0].lcao.shape[0])).T
    else:
        salc.lcao = salc.lcao.T
orbital_idxs = []
tot = 0
for h, irrep in enumerate(irreplength):
    h_idxs = []
    for i in range(tot, irrep + tot):
        h_idxs.append(i)
    orbital_idxs.append(h_idxs)
    tot += irrep
S = ints.ao_overlap().np
T = ints.ao_kinetic().np
V = ints.ao_potential().np
bigERI = ints.ao_eri().np
S = aotoso(S, salcs)

T = aotoso(T, salcs)
V = aotoso(V, salcs)
ctab = symtext.chartable
enuc = molecule.nuclear_repulsion_energy()
np.set_printoptions(threshold=sys.maxsize, linewidth=12000, precision=4)

if Settings["guess"] == "core":
    C, A, eps = rhf_core_guess(S, T, V)
elif Settings["guess"] == "gwh":
    C, A, eps = rhf_gwh_guess(S, T, V)
    #raise Exception("Sorry, gwh guess is not yet supported")

D = build_D(C, eps, ndocc)
H = T + V

bigERI = aotoso_2(bigERI, salcs)
nbfxns = psi4.core.BasisSet.nbf(basis)

#algo = 'blocky'
algo = 'blocky_sym'
#algo = 'chonky'
#algo = 'sparse'

if Settings["algo"] == "sparese":
    ERI = SparseERI(nbfxns)
    ERI.remove_asym(ctab, orbital_idxs)
    ERI.get_ints_dummy(bigERI)
    for i in range(0, 50):
        F = build_fock_sparse_sym(H, D, ERI, nbfxns)

        F = build_fock_chonky(H, D, bigERI, nbfxns)
        E = rhf_energy(D, H, F) + enuc
        print(f"Symmetry Adapted RHF Energy {E}")
        Fs = A.transpose().dot(F.dot(A))
        eps, Cs = Fs.eigh()
        C = A.dot(Cs)
        D = build_D(C, eps, ndocc)
elif algo == 'sparse':
    for i in range(0, Settings["scf_max_iter"]):
        
        F = build_fock_chonky(H, D, bigERI, nbfxns)
        
        E = rhf_energy(D, H, F) + enuc
        print(f"Symmetry Adapted RHF Energy {E}")
        Fs = A.transpose().dot(F.dot(A))
        eps, Cs = Fs.eigh()
        C = A.dot(Cs)
        D = build_D(C, eps, ndocc)
    print("finished")
    print(E)
elif algo == 'blocky':
    for i in range(0, Settings["scf_max_iter"]):
        
        F = build_fock_blocky(H, D, bigERI, nbfxns)
        
        E = rhf_energy(D, H, F) + enuc
        print(f"Symmetry Adapted RHF Energy {E}")
        Fs = A.transpose().dot(F.dot(A))
        eps, Cs = Fs.eigh()
        C = A.dot(Cs)
        D = build_D(C, eps, ndocc)
    print("finished")
    print(E)

elif algo == 'blocky_sym':
    
    #eri repack
    print("Repacking and symmetry blocking ERI")
    before = time.time()
    ERI_swapped = np.swapaxes(bigERI, 1,2)
    dpd = DPD(orbital_idxs, symtext)
    dpd.lookup_hf_ERI_J(bigERI)
    #twod pre J
    repacked_bigERI = dpd.twod_tensor
    #twod pre K
    dpd.lookup_hf_ERI_J(ERI_swapped)
    repacked_bigERI_swapped = dpd.twod_tensor
    now = time.time()
    print(f"Finished repack {now - before:6.3f}")
    print("Starting SCF Iterations")
    start = time.time()
    for i in range(0, Settings["scf_max_iter"]):
        before = time.time()
        F = build_fock_blocky_sym(H, D, repacked_bigERI, repacked_bigERI_swapped, nbfxns)
        E = rhf_energy(D, H, F) + enuc
        Fs = A.transpose().dot(F.dot(A))
        eps, Cs = Fs.eigh()
        C = A.dot(Cs)
        D = build_D(C, eps, ndocc)
        now = time.time()
        print(f"Iter {i} SCF energy {E} took {now - before:6.3f} seconds")
    finished = time.time()
    print(f"Final RHF energy {E} which took {finished - start:6.3f} seconds")
    print(E)

#compare against psi
psi4.set_options({'basis': Settings["basis"],
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'e_convergence': 1e-10,
                     'reference': 'rhf',
                     'guess' : 'core',
                     "puream": True,
                     "freeze_core": False})
ndocc = Settings["nalpha"]
nbfxns = psi4.core.BasisSet.nbf(basis)
pe, wfn = psi4.energy('scf', return_wfn=True)

print(f"Difference between us and PSI4: {abs(E-pe)}")

