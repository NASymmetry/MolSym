import numpy as np
import molsym
#from molsym.salcs.SymmetryEquivalentIC import *
from .SymmetryEquivalentIC import *
import molsym.symtext.irrep_mats as IrrepMats
np.set_printoptions(suppress=True, linewidth=12000, precision=3)
"""
SALCs for Internal Coordinate BFs. 
"""


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
                self.lcao = new_salc[None,:]
            else:
                for y in self.lcao:
                    if np.allclose(new_salc, y, atol = 1e-6): #|| isapprox(s, -y, atol = 1e-6)
                        check = False
                        break
                    elif len(self.lcao) > 1 and np.linalg.matrix_rank(np.vstack((self.lcao, new_salc)),tol = 1e-5) <= np.linalg.matrix_rank(self.lcao, tol = 1e-5):
                        check = False
                        break
                if check:
                    S = molsym.symtools.normalize(new_salc)
                    self.lcao = np.vstack((self.lcao, S))

    @property
    def shape(self):
        if self.lcao is None:
            return (0,0)
        else:
            return self.lcao.shape

def salc_irreps(ct, nbfxns):
    salcs = []
    for i in ct.irreps:
        salcs.append(SALCblock(i))#, np.zeros(nbfxns), []))
    return salcs

def is_zeros(vec):
    return np.allclose(vec, np.zeros(vec.shape), atol=1e-6)

def InternalProjectionOp(fn, coord_list, ic_types):
    #number of redundant internal coordinates
    #mol = Molecule.from_schema(qc_mol)
    mol = molsym.Molecule.from_file(fn)
    symtext = molsym.Symtext.from_molecule(mol)
    #print(f"symtext {symtext}")
    mole = symtext.mol
    
    SEAs = mole.find_SEAs()
    #loop through zmat index_dict and remove one from each indice,
    #make it a list of Internal Coordinate class instances
    IC_obj = InternalCoordinates(coord_list, mole, ic_types)

    IC_obj.run(SEAs, symtext) 
    
    numred = len(IC_obj.ic_list)
    salcs = salc_irreps(symtext.chartable, numred)
    # Ignore coord_alignment!!!
    sea_chk = []
    #loop over Symmetry Equivalent Internal Coordinate Sets
    for seicidx, seic_set in enumerate(IC_obj.SEICs):
        #if degenerate selection isn't aligned, pick new equivcoord, 
        sea_chk.append(seicidx)
        #index of the equivcoord with respect to seic_set. This needs re-written
        equivcoord = seic_set.ICindices[0]
        print(equivcoord)
        #for equivcoord in seic_set.ICindices:
        for ir, irrep in enumerate(symtext.chartable.irreps):
            irrmat = getattr(IrrepMats, "irrm_" + str(symtext.pg))[irrep]
            dim = np.array(irrmat[0]).shape[0]
            salc = np.zeros((dim, dim, numred))
            for sidx in range(len(symtext)):
                ic2 = IC_obj.ic_map[equivcoord, sidx]
                p   =     IC_obj.phase_map[equivcoord, sidx]
                salc[:,:,ic2] += (irrmat[sidx, :, :]) * p
            for i in range(dim):
                for j in range(dim):
                    salcs[ir].addlcaonew(salc[i,j,:])

            #if the irrep is degerate and the salc is valid
            sea_chk.append(seicidx)
    np.set_printoptions(threshold=np.inf, precision=4)
    irreps = []
    dims = []
    for ir, irrep in enumerate(symtext.chartable.irreps):
        irrmat = getattr(IrrepMats, "irrm_" + str(symtext.pg))[irrep]
        dim = np.array(irrmat[0]).shape[0]
        dims.append(dim)
        if salcs[ir].shape[0] == 0:
            #salcs[ir].lcao = None
            irreps.append(0)
        else:
            #salcs[ir].lcao = np.delete(salcs[ir].lcao, 0, 0)
            irreps.append(salcs[ir].lcao.shape[0])

    suum = 0
    for s in salcs:
        print(s.shape)
        suum += s.shape[0]
    print(suum)
    print(salcs[5].lcao)
    boobus = salcs[5].lcao
    print(np.dot(boobus[2,:],boobus[3,:]))