import numpy as np
from . import irrep_mats
from .main import pg_to_symels
from .multiplication_table import build_mult_table

def irrep_things(pg):
    irrm = getattr(irrep_mats, "irrm_" + str(pg))
    symels = pg_to_symels(pg)
    mtable = build_mult_table(symels)
    mtab_chks = True
    for k in irrm:
        mtab_chk = mtable_check(k, irrm[k], mtable)
        if mtab_chk == False:
            mtab_chks = False
    gchk = goat_chk(irrm)
    return mtab_chks, gchk

def mtable_check(k, irrm, mtable):
    l = mtable.shape[0]
    for i in range(l):
        for j in range(l):
            if mtable[i,j] in multifly(irrm, i, j):
                pass
            else:
                print(f"Irrep. {k}\nMat. 1: {irrm[i]}\nMat. 2: {irrm[j]}")
                print(f"Multiplying {i} and {j}")
                print(multifly(irrm, i, j, printem=False))
                return False
    return True

def multifly(irrm, a, b, printem=False):
    l = len(irrm)
    out = []
    errl = []
    for i in range(l):
        if irrm[a].shape[0] == 1:
            r = [irrm[a][0]*irrm[b][0]]
        else:
            r = np.dot(irrm[a],irrm[b])
        errl.append(r)
        if printem:
            print(irrm[i])
            print(r)
        if np.isclose(irrm[i], r, atol = 1e-12).all():
            out.append(i)
    return out

def goat_chk(irrm):
    return_full_chk = False
    l = len(irrm)
    gc_final = []
    for mu in irrm:
        for nu in irrm:
            g = len(irrm[mu])
            d1 = irrm[mu][0].shape[0]
            d2 = irrm[nu][0].shape[0]
            if return_full_chk:
                gc = np.zeros(d1,d2,d1,d2)
            for i in range(d1):
                for j in range(d2):
                    for l in range(d1):
                        for m in range(d2):
                            gchk = 0
                            for r in range(len(irrm[mu])):
                                gchk += np.conj(irrm[mu][r][i,l])*irrm[nu][r][j,m]
                            #gchk = goat_chk(irrm[μ], irrm[ν], i,j,l,m)
                            if mu == nu and i == j and l == m:
                                expected = g / d1
                            else:
                                expected = 0
                            if return_full_chk:
                                gc[i,j,l,m] = gchk
                            elif np.isclose(gchk, expected, atol=1e-12):
                                continue
                            else:
                                #return false
                                raise(Exception(f"GOAT Check failed for irreps {mu} and {nu}, and indices {i},{l},{j},{m}. Returned value {gchk}. Expected value {expected}"))
            if return_full_chk:
                gc_final.append([mu,nu,gc])
    if return_full_chk:
        return gc_final
    else:
        return True
