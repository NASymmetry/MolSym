import numpy as np
from .symtext import *
import itertools
import re

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
    sigma_h = False
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
        elif symel.symbol[:7] == "sigma_h":
            sigma_h = True
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
                if sigma_h:
                    pg = f"D{highest_Cn}h"
                else:
                    pg = f"D{highest_Cn}d"
            elif sigma_h:
                pg = f"D{highest_Cn}h"
            elif sigma:
                pg = f"D{highest_Cn}d"
            else:
                pg = f"D{highest_Cn}"
        elif sigma_h:
            pg = f"C{highest_Cn}h"
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

def subgroups(symels, mtable, restrict_comb=None):
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

    possible_subgroup_orders = divisors(h)[1:-1] #Exclude 1 and h
    #print(unique_cycles)
    
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
                subgroups.append(list(subgroup_candidate))
                subgroup_pgs.append(identify_subgroup(subgroup_candidate, symels))
    return subgroups, subgroup_pgs