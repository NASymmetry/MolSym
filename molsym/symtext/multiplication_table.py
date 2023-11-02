import numpy as np
import itertools
import re
import math
from . import irrep_mats

def multifly(symels, A, B):
    Crrep = np.dot(A.rrep,B.rrep)
    for (i,g) in enumerate(symels):
        if np.isclose(Crrep, g.rrep).all():
            return i,g
    raise Exception(f"No match found for Symels {A.symbol} and {B.symbol}!")

def build_mult_table(symels):
    h = len(symels)
    t = np.zeros((h,h), dtype=int)
    for (i,a) in enumerate(symels):
        for (j,b) in enumerate(symels):
            t[i,j] = multifly(symels, a, b)[0]
    return t

def divisors(n):
    # This isn't meant to handle large numbers, thankfully most point groups have an order less than 100
    out = []
    for i in range(n):
        if n % (i+1) == 0:
            out.append(i+1)
    return out

def is_subgroup(G, mtable):
    # Testing for closure as all other group properties are enforced elsewhere
    for ga in G:
        for gb in G:
            if mtable[ga,gb] in G:
                continue
            else: return False
    return True

def identify_subgroup(subgroup, symels):
    subgroup_symels = [symels[i] for i in subgroup]
    inversion = False
    highest_Cn = None
    #sigma_h = False
    sigma = False
    Sn = False
    mult_C2s = False
    one_C2 = False
    nC3 = 0
    nC4 = 0
    n_re = re.compile(r"C_(\d*)\^?\d*")
    sigma_vd_re = re.compile(r"sigma_[vd]")
    for symel in subgroup_symels:
        if symel.symbol == "i":
            inversion = True
        #elif symel.symbol[:7] == "sigma_h":
        #    sigma_h = True
        #elif re.match(sigma_vd_re, symel.symbol):
        elif symel.symbol[:5] == "sigma":
            sigma = True
        elif symel.symbol[0] == "S":
            Sn = True
        elif symel.symbol[0] == "C":
            if symel.symbol[:3] == "C_2":
                if one_C2:
                    mult_C2s = True
                else:
                    one_C2 = True
            m = re.match(n_re, symel.symbol)
            if m:
                n = m.groups()[0]
            else:
                n = None
            if n is not None:
                n = int(n)
                if n == 3:
                    nC3 +=1
                elif n == 4:
                    nC4 += 1
                if highest_Cn is None:
                    highest_Cn = n
                if highest_Cn < n:
                    highest_Cn = n
    if highest_Cn is not None:
        highest_Cn_even = (highest_Cn % 2) == 0
        if nC3 == 20:
            if inversion:
                pg = "Ih"
            else:
                pg = "I"
        elif nC3 == 8:
            if nC4 == 6:
                if inversion:
                    pg = "Oh"
                else:
                    pg = "O"
            else:
                if inversion:
                    pg = "Th"
                elif Sn:
                    pg = "Td"
                else:
                    pg = "T"
        elif mult_C2s:
            if inversion:
                if highest_Cn_even:
                    pg = f"D{highest_Cn}h"
                else:
                    pg = f"D{highest_Cn}d"
            elif highest_Cn_even and Sn:
                pg = f"D{highest_Cn}d"
            elif Sn:
                pg = f"D{highest_Cn}h"
            else:
                pg = f"D{highest_Cn}"
        elif sigma:
            if inversion or Sn:
                pg = f"C{highest_Cn}h"
            else:
                pg = f"C{highest_Cn}v"
        elif Sn:
            pg = f"S{highest_Cn*2}"
        elif sigma:
            pg = f"C{highest_Cn}v"
        else:
            pg = f"C{highest_Cn}"
    else:
        if inversion:
            pg = "Ci"
        elif sigma:
            pg = "Cs"
        else:
            pg = "C1"

    return pg

def cycles(symels, mtable):
    # Find cycles of each element, only consider combinations of cycles
    h = len(symels)
    cycles = []
    for gi in range(1,h):
        cycle_i = [gi]
        gold = gi
        while True:
            gnew = mtable[gold, gi]
            cycle_i.append(gnew)
            if gnew == 0:
                cycles.append(set(cycle_i))
                break
            gold = gnew
    
    # Reduce to unique cycles
    unique_cycles = [cycles[0]]
    for ci in cycles:
        chk = True
        for uci in unique_cycles:
            if ci == uci:
                chk = False
                break
        if chk:
            unique_cycles.append(ci)
    return unique_cycles

def subgroups(symels, mtable, restrict_comb=None):
    # Naive implementation of subgroup search (by combinations of cycles)
    h = len(symels)
    unique_cycles = cycles(symels, mtable)
    possible_subgroup_orders = divisors(h)[1:-1] #Exclude 1 and h
    
    subgroups = []
    subgroup_pgs = []
    # Select n sets of unique cycles. Each cycle should be a subgroup, but all cycles comprise the group so skip
    if restrict_comb is not None:
        nselect_limit = restrict_comb
    else:
        nselect_limit = len(unique_cycles)-1
    for nselect in range(1,nselect_limit): 
        itercomb = itertools.combinations(unique_cycles, nselect)
        combined = []
        for comb_i in itercomb:
            s0 = comb_i[0]
            for si in range(1,len(comb_i)):
                s0 = s0.union(comb_i[si])
            combined.append(s0)
        #print(combined)
        for subgroup_candidate in combined:
            if subgroup_candidate in subgroups:
                continue
            if len(subgroup_candidate) not in possible_subgroup_orders:
                continue
            if is_subgroup(subgroup_candidate, mtable):
                accepted_subgroup = list(subgroup_candidate)
                accepted_subgroup.sort()
                subgroups.append(accepted_subgroup)
                subgroup_pgs.append(identify_subgroup(accepted_subgroup, symels))
    return subgroups, subgroup_pgs

def multiply(mtable, *args):
    m = args[0]
    for mi in args[1:]:
        m = mtable[m, mi]
    return m

def subgroups_better(symels, mtable, max_subgroup_size=None, restrict_comb=None):
    h = len(symels)
    unique_cycles = cycles(symels, mtable)
    possible_subgroup_orders = divisors(h)[1:-1] #Exclude 1 and h
    if max_subgroup_size is not None:
        mss = max_subgroup_size
    else:
        mss = h
    if restrict_comb is not None:
        nselect_limit = restrict_comb
    else:
        nselect_limit = len(unique_cycles)-1
    subgroups = [[*i] for i in unique_cycles]
    subgroup_pgs = [identify_subgroup(i, symels) for i in unique_cycles]
    # Select n sets of unique cycles. Each cycle should be a subgroup, but all cycles comprise the group so skip
    for nselect in range(2,nselect_limit+1): 
        itercomb = itertools.combinations(unique_cycles, nselect)
        for comb_i in itercomb:
            s = math.prod([len(cycle) for cycle in comb_i])
            if s > mss or s >= h or s not in possible_subgroup_orders:
                continue
            product = itertools.product(*comb_i)
            grp = []
            for p in product:
                grp.append(multiply(mtable, *p))
            grp = [*set(grp)]
            grp.sort()
            if grp not in subgroups and is_subgroup(grp, mtable):
                subgroups.append(grp)
                subgroup_pgs.append(identify_subgroup(grp, symels))
    return subgroups, subgroup_pgs

def mtable_check(irrm, mtable):
    l = mtable.shape[0]
    for irrep in irrm:
        for i in range(l):
            for j in range(l):
                if mtable[i,j] in irrm_multifly(irrm[irrep], i, j):
                    continue
                else:
                    #print(f"Irrep. {irrep}\nMat. 1: {irrm[i]}\nMat. 2: {irrm[j]}")
                    #print(f"Multiplying {i} and {j}")
                    #print(irrm_multifly(irrm, i, j))
                    return False
    return True

def irrm_multifly(irrm, a, b):
    l = irrm.shape[0]
    out = []
    errl = []
    for i in range(l):
        if irrm[a].shape[0] == 1:
            r = [irrm[a][0]*irrm[b][0]]
        else:
            r = irrm[a]*irrm[b]
        errl.append(r)
        if np.isclose(irrm[i], r, atol = 1e-10).all:
            out.append(i)
    return out

def orient_subgroup_to_irrmat(subgroup_symels, subgroup_pg):
    irrm = eval(f"irrep_mats.irrm_{subgroup_pg}")
    subgroup_mtable = build_mult_table(subgroup_symels)
    # Assume E is always first
    perm_idxs = itertools.permutations(range(1,len(subgroup_symels)))
    for p in perm_idxs:
        subgroup_perm_idx = [0] + list(p)
        subgroup_perm = [subgroup_symels[i] for i in subgroup_perm_idx]
        if mtable_check(irrm, subgroup_mtable[np.ix_(subgroup_perm_idx, subgroup_perm_idx)]):
            return subgroup_perm
    return False