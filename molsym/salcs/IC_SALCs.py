import numpy as np
import molsym
from .SymmetryEquivalentIC import *
import molsym.symtext.irrep_mats as IrrepMats
import warnings
np.set_printoptions(suppress=True, linewidth=12000, precision=3)
"""
SALCs for Internal Coordinate BFs. 
"""

"""
Deprecated!
"""

class SALCblock():
    def __init__(self, irrep) -> None:
        # Init with no SALCs
        warnings.warn("Using deprecated code")
        self.irrep = irrep
        self.lcao = None
        if self.irrep[0] == "A" or self.irrep[0] == "B":
            self.pfxn_idxs = None
        else:
            self.pfxn_idxs = []

    def append(self, new):
        if self.lcao is None:
            # If no SALCs in lcao, initialize self.lcao
            self.lcao = new
        else:
            # If lcao initialized, append to self.lcao
            self.lcao = np.vstack((self.lcao, new))

    def addlcaonew(self, new_salc, pfxnidx):
        if not is_zeros(new_salc):
            check = True
            if self.lcao is None:
                if self.pfxn_idxs is not None:
                    self.pfxn_idxs.append(pfxnidx)
                self.lcao = molsym.symtools.normalize(new_salc[None,:])
            else:
                if self.lcao.shape[0] == 1:
                    rank = 1
                else:
                    rank = np.linalg.matrix_rank(self.lcao, tol=1e-5)
                if np.linalg.matrix_rank(np.vstack((self.lcao, new_salc)),tol = 1e-5) <= rank:
                    check = False
                if check:
                    if self.pfxn_idxs is not None:
                        self.pfxn_idxs.append(pfxnidx)
                    S = molsym.symtools.normalize(new_salc)
                    self.lcao = np.vstack((self.lcao, S))

    @property
    def shape(self):
        if self.lcao is None:
            return (0,0)
        else:
            return self.lcao.shape

def salc_irreps(ct, nbfxns):
    warnings.warn("Using deprecated code")
    salcs = []
    for i in ct.irreps:
        salcs.append(SALCblock(i))#, np.zeros(nbfxns), []))
    return salcs

def is_zeros(vec):
    warnings.warn("Using deprecated code")
    return np.allclose(vec, np.zeros(vec.shape), atol=1e-6)

def span(ics, symtext, icmap):
    warnings.warn("Using deprecated code")
    chorker_loaf = np.zeros(icmap.shape)
    for i in range(icmap.shape[0]):
        for j in range(icmap.shape[1]):
            if icmap[i,j] == i:
                chorker_loaf[i,j] = 1
    rhorker_loaf = chorker_loaf[ics.ICindices, :]
    shorker_loaf = np.sum(rhorker_loaf, axis=0)
    rshorker_loaf = np.zeros(symtext.chartable.characters.shape[1])
    for i in range(len(symtext)):
        rshorker_loaf[symtext.class_map[i]] = shorker_loaf[i]
    span = np.zeros(len(symtext.chartable.irreps), dtype=np.int32)
    for idx, irrep in enumerate(symtext.chartable.irreps):
        n = round(np.sum(rshorker_loaf * symtext.chartable.class_orders * symtext.chartable.characters[idx,:]) / symtext.order)
        span[idx] = n
    return span

def InternalProjectionOp(fn, coord_list, ic_types):
    #number of redundant internal coordinates
    #mol = Molecule.from_schema(qc_mol)
    warnings.warn("Using deprecated code")
    mol = molsym.Molecule.from_file(fn)
    symtext = molsym.Symtext.from_molecule(mol)
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
    print(IC_obj.SEICs)
    #loop over Symmetry Equivalent Internal Coordinate Sets
    for seicidx, seic_set in enumerate(IC_obj.SEICs):
        #if degenerate selection isn't aligned, pick new equivcoord, 
        sea_chk.append(seicidx)
        #index of the equivcoord with respect to seic_set. This needs re-written
        equivcoord = seic_set.ICindices[0]
        spn = span(seic_set, symtext, IC_obj.ic_map)
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
                    salcs[ir].addlcaonew(salc[i,j,:], i)

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
        print(f"{s.shape[0]} {s.irrep}")
        print(s.pfxn_idxs)
        suum += s.shape[0]
    print(suum)
    
    # rCC (0-1) by aHCC (6-9) for E' (2)
    irrep = 2
    rCC = 1
    aHCC = 6
    print(salcs[irrep].lcao)
    blue = [[0,0],[0,1],[0,2],[0,3],[1,8],[1,9],[1,10],[1,11],[2,4],[2,5],[2,6],[2,7]]
    red = [[0,4],[0,6],[0,8],[0,10],[1,1],[1,3],[1,5],[1,7],[2,0],[2,2],[2,9],[2,11]]
    green = [[0,5],[0,7],[0,9],[0,11],[1,0],[1,2],[1,4],[1,6],[2,1],[2,3],[2,8],[2,10]]
    for aHCC in range(6,10):
        print(aHCC)
        thissun = np.outer(salcs[irrep].lcao[rCC,0:3], salcs[irrep].lcao[aHCC,12:])
        print(thissun)
        bout = 0
        rout = 0
        gout = 0
        for b in blue:
            bout += thissun[b[0],b[1]]
        for r in red:
            rout += thissun[r[0],r[1]]
        for g in green:
            gout += thissun[g[0],g[1]]
        print(bout, rout, gout)