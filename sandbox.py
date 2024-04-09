from molsym.symtext.general_irrep_mats import *
import numpy as np
np.set_printoptions(precision=3, threshold=np.inf, linewidth=14000, suppress=True, sign=' ', floatmode="fixed")
#symels, irreps, irrep_mats = Zn(5,"C")
#i = Symel("i", None, -1*np.eye(3), 0, 0, "i")
#sh = Symel("sigma_h", np.array([0,0,1]), np.array([[1,0,0],[0,1,0],[0,0,-1]]), 0, 0, "sigma_h")
#print(direct_product(symels, irreps, irrep_mats, sh, "Empty"))

#print(pg_to_symels("C4v"))

symels, irreps, irrep_mat = pg_to_symels("D2d")
for symel in symels:
    print(symel.symbol)
    print(symel.vector)
    print(symel.rrep)

def doodoo(pg):
    symels, irreps, irrep_mat = pg_to_symels(pg)
    #for symel in symels:
    #    print(symel.rrep)
    #
    #for irrep in irreps:
    #    print(irrep_mat[irrep.symbol].reshape((len(symels), irrep.d**2)))
    #print(irrep_mat)
    print(irreps)
    print(symels)
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
aa = [f"D{i}d" for i in range(2,9)]
#aa = ["C4v"]
#for a in aa:
#    print(a)
#    doodoo(a)