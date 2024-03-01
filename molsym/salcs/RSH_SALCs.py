import psi4
import numpy as np
from molsym.salcs.RSHoperations import generateRotations
#from RSHoperations import generateRotations
#import san_diego
from molsym.symtext import symtext, symel_generators
#from molecule import Molecule
from dataclasses import dataclass
#import IrrepMats
import molsym.symtext.irrep_mats as IrrepMats
from numpy import linalg as LA
import molsym

np.set_printoptions(suppress=True, linewidth=12000, precision=4)

"""
SALCs for RSH and potentially cartesian BFs. Each atom of 
charge Z in the molecule must have the same basis functions.
"""

#return a list of am associated with each atom 
#in the molecule. 
def get_basis(molecule, basis):
    nbas_vec = []
    molecule_basis = []
    counter = 0
    for x in range(0, molecule.natom()):
        atom_basis = []
        for y in range(0, basis.nshell_on_center(x)):
            atom_basis.append(basis.shell(y+counter).am)
            #print(basis.shell(y+counter).am)
        counter += basis.nshell_on_center(x)
        L = 0
        for l in atom_basis:
            L += 2*l + 1
        molecule_basis.append(atom_basis)
        nbas_vec.append(L)
    return molecule_basis, nbas_vec


#count the number of shells between an atom
def obstruct(atom1, atom2, nbas_vec):
    if atom1 == atom2:
        obstruction = 0
    else:
        obstruction = 0
    
        A = int(atom1 + 1)
        B = int(atom2)
        for x in nbas_vec[A:B]:
            obstruction += x
    return obstruction

#collect parameterized rsh rotations for each symmetry operation
def rotate_em(maxam, ops):
   rsh_rot_per_op = []
   for i, op in enumerate(ops):
       rsh_rot_per_op.append(generateRotations(maxam, op.rrep))
   return rsh_rot_per_op

class SALCblock():
    def __init__(self, irrep) -> None:
        # Init with no SALCs
        self.irrep = irrep
        self.lcao = None
    
    def append(self, new):
        if self.lcao is None:
            # If no SALCs in lcao, initialize self.lcao
            self.lcao = new
        else:
            # If lcao initialized, append to self.lcao
            self.lcao = np.vstack((self.lcao, new))


    def addlcaonew(self, new_salc):
        if not is_zeros(new_salc):
            check = True
            if self.lcao is None:
                self.lcao = molsym.symtools.normalize(new_salc[None,:])
            else:
                if self.lcao.shape[0] == 1:
                    rank = 1
                else:
                    rank = np.linalg.matrix_rank(self.lcao, tol=1e-5)
                if np.linalg.matrix_rank(np.vstack((self.lcao, new_salc)),tol = 1e-5) <= rank:
                    check = False
                if check:
                    S = molsym.symtools.normalize(new_salc)
                    self.lcao = np.vstack((self.lcao, S))
    @property
    def shape(self):
        if self.lcao is None:
            return (0,0)
        else:
            return self.lcao.shape


def is_zeros(vec):
    return np.allclose(vec, np.zeros(vec.shape), atol=1e-6)

def normalize(B):
    return B / LA.norm(B)

def salc_irreps(ct, nbfxns):
    salcs = []
    for i in ct.irreps:
        salcs.append(SALCblock(i))#, np.zeros(nbfxns), []))
    return salcs
def psi4_order(nbas_vec, outers):
    vec = []
    for out in outers:
        for l in out:
            #print(f"l {l}")
            if l == 0:
                #print("no change")
                vec.append([0])
            else:
                count = 0
                beeb = [x for x in range(0, l*2 + 1)]
                for ind in range(0, l + 1):
                    squeeb = beeb[ind - count]
                    beeb = np.delete(beeb, ind - count)
                    beeb = np.insert(beeb, 2 * l - ind - count, squeeb)
                    count += 1
                beeb = list(beeb)
                vec.append(beeb)
    
    #vec = np.array([item for sublist in vec for item in sublist])  
    vec = [item for sublist in vec for item in sublist]  
    beeb = [x for x in range(0, sum(nbas_vec))]
    total = 0
    PP = []
    for nbas in outers:
        for n in nbas:
            l = 2 * n + 1
            squeeb = beeb[total:l + total]
            psi = vec[total:l + total]
            pp = [squeeb[i] for i in psi]
                
            PP.append(pp)
            total += l
    doit = True
    if doit:
        PP = [item for sublist in PP for item in sublist]
    else:
        PP = [x for x in range(0, sum(nbas_vec))]
    return PP
def Project(mole, mool, bset, symtext):
    #temp seas
    symm_equiv = []
    done = []
    for atom_i in range(len(symtext.mol)):
        if atom_i in done:
            continue
        equiv_set = []
        for sidx in range(len(symtext)):
            newatom = symtext.atom_map[atom_i, sidx]
            equiv_set += [newatom]
        reduced_equiv_set = list(set(equiv_set))
        symm_equiv.append(reduced_equiv_set)
        done += reduced_equiv_set
    #seas = mole.find_SEAs()
    nbfxns = psi4.core.BasisSet.nbf(bset)
    outers, nbas_vec = get_basis(mool, bset)
    reorder_psi4 = psi4_order(nbas_vec, outers)
    maxam = psi4.core.BasisSet.max_am(bset)
    g = psi4.core.BasisSet.nshell(bset)
    nbf = psi4.core.BasisSet.nbf(bset)
    #print("Generating symmetry operations in RSH basis")
    #print(symtext.symels)
    rotated = rotate_em(maxam, symtext.symels)
    salcs = salc_irreps(symtext.chartable, nbfxns)
    basis = 0
    sea_chk = []
    #loop over atom index
    for atomidx, atom in enumerate(mole):
        #for seaidx, sea in enumerate(seas):
        for seaidx, sea in enumerate(symm_equiv):
            #if atomidx in sea.subset:
            if atomidx in sea:
                if seaidx in sea_chk:
                    basis += nbas_vec[atomidx]
                else:
                    sea_chk.append(seaidx)
                    #equivatom = seas[seaidx].subset[0]
                    equivatom = symm_equiv[seaidx][0]
                    bsfxn_counter = 0
                    #loop over l value in equivalent atom basis
                    for k, l in enumerate(outers[equivatom]):
                        for ml in range(0,2*l + 1):
                            bsfxn_counter += 1
                            #loop over irreps of ctab
                            #grab rotation slice, 1 x 2*l + 1 shape
                            #broadcast car * slice
                            #print(f"l ml {l} {ml}")
                            for ir, irrep in enumerate(symtext.chartable.irreps):
                                irrmat = getattr(IrrepMats, "irrm_" + str(symtext.pg))[irrep]
                                #irrmat = Irrmat[irrep]
                                #print("the irrmat")
                                #print(irrmat)
                                dim = np.array(irrmat[0]).shape[0]
                                salc = np.zeros((dim, dim, nbfxns))
                                salc = projection(salc, bset, symtext, irrep, irrmat, rotated, l, ml, equivatom, basis, nbas_vec, sea)
                                #print(f"ir {ir} {salc}")
                                #print(salc)
                                #salc = salc[:, :,reorder_psi4]
                                #print(salc)
                                chk = False
                                for i in range(dim):
                                    for j in range(dim):
                                        salcs[ir].addlcaonew(salc[j,i,:])
                                #jprint("We might have added a salc")
                                #jprint(salcs[ir].lcao)
                                #salcs, chk = addlcaonew(salcs, salc, ir, irrep)                            
                                
                                #if chk:
                                #    print(f"{irrep} {salc}")
                        basis += 2* l +1

    irreps = []
    for s, salc in enumerate(salcs):
        if salc.lcao is None:
            irreps.append(0)
        else:
            irreps.append(salc.lcao.shape[0])
    #for salc in salcs:
    #    print(salc.lcao)
    return salcs, irreps, reorder_psi4
def projection(salc, bset, symtext, irrep, irrmat, rotated, l, ml, equivatom, basis, nbas_vec, sea):
    dim = np.array(irrmat[0]).shape[0]
    for op, Class in enumerate(symtext.class_map):
        rotate_result = rotated[op]
        atom2 = symtext.atom_map[:,op][equivatom]
        if l == 0:
            result = np.eye(1)
        else:
            result = rotate_result[l][:, ml]
        result = np.squeeze(result)
        
        atom2 = symtext.atom_map[:,op][equivatom]
        if atom2 == equivatom:
            for i in range(0, dim):
                for j in range(0, dim):
                    salc[i, j,basis:basis + (2 * l) + 1] += (irrmat[op,i,j] * result)
        else:
            obstruction = obstruct(equivatom, atom2, nbas_vec)
            offset = obstruction + nbas_vec[equivatom]
            for i in range(0, dim):
                for j in range(0, dim):
                    salc[i,j,basis + offset:basis + offset + (2 * l) + 1] += irrmat[op,i,j]*result
            
    for ind, s in enumerate(salc):
        s *= (symtext.chartable.irrep_dims[irrep] / symtext.order)
        salc[ind] = s
    return salc