from molsym.symtext.general_irrep_mats import *
import numpy as np
np.set_printoptions(precision=3, threshold=np.inf, linewidth=14000, suppress=True, sign=' ', floatmode="fixed")
#symels, irreps, irrep_mats = Zn(5,"C")
#i = Symel("i", None, -1*np.eye(3), 0, 0, "i")
#sh = Symel("sigma_h", np.array([0,0,1]), np.array([[1,0,0],[0,1,0],[0,0,-1]]), 0, 0, "sigma_h")
#print(direct_product(symels, irreps, irrep_mats, sh, "Empty"))

#print(pg_to_symels("C4v"))

#symels, irreps, irrep_mat = pg_to_symels("D2h")
#for symel in symels:
#    print(symel.symbol)
##    print(symel.vector)
##    print(symel.rrep)
##print(irreps)
#from molsym.symtext.multiplication_table import build_mult_table
#print(build_mult_table(symels))

import molsym
from molsym.symtext.general_irrep_mats import pg_to_symels
#from molsym.symtext.symtext_helper import get_class_name
#symels, irreps, irrep_mats = pg_to_symels("Ih")

mol = molsym.Molecule.from_file("test/xyz/benzene.xyz")
v = molsym.symtools.normalize(np.random.uniform(size=3))
#theta = 2*np.pi * np.random.uniform()
#randperm = np.random.permutation(len(mol))
#mol.coords[:,:] = mol.coords[randperm,:]
#mol = mol.transform(molsym.symtools.rotation_matrix(v, theta))
mol = molsym.symmetrize(mol)
s = molsym.Symtext.from_molecule(mol)
print(s.pg)
print(s.character_table)
print(s.classes)
print(s.symel_to_class_map)
print(s.class_orders)

#dp = s.direct_product(*[0,1,2])
#print(dp)
#print(s.reduction_coefficients(dp))

#import molsym
#mol = molsym.Molecule.from_file("acetylene.xyz")
#mol = molsym.symmetrize(mol)
#s = molsym.Symtext.from_molecule(mol)
#fxn_list = [[[0,1],"R1"],[[0,2],"R2"],[[1,3],"R3"],[[1,0,2],"LinX1"],[[1,0,2],"LinY1"],[[0,1,3],"LinX2"],[[0,1,3],"LinY2"]]
#from molsym.salcs.linear_functions import LinearInternalCoordinates
#fxn_set = LinearInternalCoordinates(s, fxn_list)
#
#print(fxn_set.partners)
#print(fxn_set.fxn_map)
#print(fxn_set.SE_fxns)

def doodoo(pg):
    symels, irreps, irrep_mat = pg_to_symels(pg)
    #for symel in symels:
    #    print(symel.rrep)
    #
    #for irrep in irreps:
    #    print(irrep_mat[irrep.symbol].reshape((len(symels), irrep.d**2)))
    #print(irrep_mat)
    #print(irreps)
    #print(symels)
    from molsym.symtext.goat import goat_chk, mtable_check
    from molsym.symtext.multiplication_table import build_mult_table

    mtable = build_mult_table(symels)
    mtab_chks = True
    for k in irrep_mat:
        mtab_chk = mtable_check(k, irrep_mat[k], mtable)
        if mtab_chk == False:
            mtab_chks = False
    gchk = goat_chk(irrep_mat)
    print(mtab_chk, gchk)

#aa = ["C1", "Ci", "Cs"] + [f"C{i}" for i in range(9)]
#aa = [f"D{i}d" for i in range(2,9)]
#doodoo("Ih")
#aa = ["C4v"]
#for a in aa:
#    print(a)
#    doodoo(a)