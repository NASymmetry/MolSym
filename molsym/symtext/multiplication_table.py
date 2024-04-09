import numpy as np
import itertools
import re
#from .symel import pg_to_symels
from molsym.symtools import *
from .general_irrep_mats import pg_to_symels

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

def cycles(mtable):
    # Find cycles of each element
    h = mtable.shape[0]
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

def multiply(mtable, *args):
    m = args[0]
    for mi in args[1:]:
        m = mtable[m, mi]
    return m

def subgroup_by_name(symels, mult_table, subgroup):
    if subgroup == "C1":
        return [[0,0]]
    subgroup_symels, irreps, irrep_mat = pg_to_symels(subgroup)
    subgroup_mult_table = build_mult_table(subgroup_symels)
    big_list = []
    for sg_el in range(len(subgroup_symels)):
        small_list = []
        for el in range(len(symels)):
            if same_type(symels[el], subgroup_symels[sg_el]):
                small_list.append(el)
        big_list.append(small_list)
    subgroup_cycle_set = cycles(subgroup_mult_table)
    min_subgroup_cycle_set = min_cycle_set(subgroup_mult_table, subgroup_cycle_set)
    min_subgroup_cycle_generators = []
    for m in min_subgroup_cycle_set:
        cycle_i = subgroup_cycle_set[m]
        cycle_i_gen = 0
        for el in cycle_i:
            if order(subgroup_mult_table, el) == len(cycle_i):
                cycle_i_gen = el
        if cycle_i_gen == 0:
            raise(Exception("Fail"))
        min_subgroup_cycle_generators.append(cycle_i_gen)
    a = [len(big_list[i]) for i in min_subgroup_cycle_generators]
    c = itertools.product(*[range(i) for i in a])
    for dp in c:
        subgroup_isomorphism = []
        for msc_idx, min_subgroup_cycle in enumerate(min_subgroup_cycle_set):
            cycle_i_gen = min_subgroup_cycle_generators[msc_idx]
            # Assign subgroup cycle generator to similar element in full group
            subgroup_isomorphism.append([cycle_i_gen, big_list[cycle_i_gen][dp[msc_idx]]])
            # Fill out cycles
            old = big_list[cycle_i_gen][dp[msc_idx]]
            subgroup_old = cycle_i_gen
            for _idx in range(1, order(subgroup_mult_table, cycle_i_gen)):
                new = mult_table[old,big_list[cycle_i_gen][dp[msc_idx]]]
                subgroup_new = subgroup_mult_table[subgroup_old,cycle_i_gen]
                subgroup_isomorphism.append([subgroup_new, new])
                old = new
                subgroup_old = subgroup_new
        subgroup_isomorphism = sorted(subgroup_isomorphism)
        subgroup_isomorphism = [subgroup_isomorphism[i] for i in range(len(subgroup_isomorphism)) if i == 0 or subgroup_isomorphism[i] != subgroup_isomorphism[i-1]]
        
        # Fill out rest of mtable
        new = []
        norman = itertools.permutations(subgroup_isomorphism, len(min_subgroup_cycle_set))
        for norman_idx in norman:
            new.append([multiply(subgroup_mult_table, *[i[0] for i in norman_idx]), multiply(mult_table, *[i[1] for i in norman_idx])])
        subgroup_isomorphism += new
        subgroup_isomorphism = sorted(subgroup_isomorphism)
        subgroup_isomorphism = [subgroup_isomorphism[i] for i in range(len(subgroup_isomorphism)) if i == 0 or subgroup_isomorphism[i] != subgroup_isomorphism[i-1]]
        
        # Check for correct group, bijectivity, and group closure
        ham = [subgroup_isomorphism[i][1] for i in range(len(subgroup_isomorphism))]
        if identify_subgroup(ham, symels) == subgroup and len(set(ham)) == len(ham) and is_subgroup(ham, mult_table):
            s = mult_table[:,ham]
            s = s[ham,:]
            return subgroup_isomorphism

def same_type(symel_a, symel_b):
    # E, i, sigma, C_n, S_n
    rgx = re.compile(r"_(\d*)\^?")
    if symel_a.symbol[0] == "E" and symel_b.symbol[0] == "E":
        return True
    elif symel_a.symbol[0] == "i" and symel_b.symbol[0] == "i":
        return True
    elif symel_a.symbol[0:5] == "sigma" and symel_b.symbol[0:5] == "sigma":
        return True
    elif symel_a.symbol[0] == "C" and symel_b.symbol[0] == "C":
        if rgx.search(symel_a.symbol).groups()[0] == rgx.search(symel_b.symbol).groups()[0]:
            return True
    elif symel_a.symbol[0] == "S" and symel_b.symbol[0] == "S":
        if rgx.search(symel_a.symbol).groups()[0] == rgx.search(symel_b.symbol).groups()[0]:
            return True
    else:
        return False

def min_cycle_set(mult_table, cycle_set):
    h = mult_table.shape[0]
    min_set = []
    already_in = [0]
    for c_idx, c in enumerate(cycle_set):
        new = []
        for i in c:
            for j in already_in:
                new.append(mult_table[i][j])
        if set(new+already_in) == set(already_in):
            pass
        else:
            min_set.append(c_idx)
        already_in = list(set(new+already_in))
        if len(already_in) == h:
            return min_set
        elif len(already_in) > h:
            raise(Exception("Too many elements added to group"))
        
def order(mult_table, idx):
    # Returns order of element in group
    n = 0
    old = idx
    while True:
        new = mult_table[old][idx]
        n += 1
        if new == idx:
            return n
        old = new

def subgroup_axes(subgroup, symels):
    rgx = re.compile(r"_(\d*)\^?")
    paxis = np.array([0,0,0])
    saxis = np.array([0,0,0])
    
    # Cubic groups, UNTESTED TODO
    if subgroup[0] == "T":
        c2s = []
        for s in symels:
            if s.symbol[0] == "C" and int(rgx.search(s.symbol).groups()[0]) == 2:
                c2s.append(s)
        # Any two C2 axes
        paxis = c2s[0].vector
        saxis = c2s[1].vector
    elif subgroup[0] == "O":
        c4s = []
        for s in symels:
            if s.symbol[0] == "C" and int(rgx.search(s.symbol).groups()[0]) == 4:
                c4chk = True
                for c4 in c4s:
                    if issame_axis(c4.vector, s.vector):
                        c4chk = False
                if c4chk:
                    c4s.append(s)
                c4s.append(s)
        # Any two C2 axes
        paxis = c4s[0].vector
        saxis = c4s[1].vector
    elif subgroup[0] == "I":
        # Find a C2
        for s in symels:
            if s.symbol[0] == "C" and int(rgx.search(s.symbol).groups()[0]) == 2:
                c2 = s
                break
        # paxis is any C2 axis, saxis is defined as the C2 axis that is coplanar with paxis and the nearest C3 axis to paxis
        paxis = c2.vector
        c3s = []
        for s in symels:
            if s.symbol[0] == "C" and int(rgx.search(s.symbol).groups()[0]) == 3:
                c3chk = True
                for c3 in c3s:
                    if issame_axis(c3.vector, s.vector):
                        c3chk = False
                if c3chk:
                    c3s.append(s)
                c3s.append(s)
        for c3 in c3s:
            if np.isclose(np.arccos(abs(np.dot(c3, paxis))), 0.36486382647383764, abs_tol = 1e-4):
                taxis = normalize(np.cross(c3, paxis))
                saxis = normalize(np.cross(taxis,paxis))
                break
        # Reorienting vectors such that one face is on the z-axis with "pentagon" pointing at the negative y-axis
        phi = (1+np.sqrt(5.0))/2
        # Weirdness here, negative or positive???
        theta = np.arccos(phi/np.sqrt(1+(phi**2)))
        rmat = rotation_matrix(saxis, theta)
        paxis = np.dot(rmat, paxis)
        taxis = np.dot(rmat, paxis)

    # All other groups
    largest_Cn = None
    n = 0
    any_sigma = None
    sigma_chk = False
    for s in symels:
        if s.symbol[0] == "C":
            if int(rgx.search(s.symbol).groups()[0]) > n:
                largest_Cn = s
                n = int(rgx.search(s.symbol).groups()[0])
        elif s.symbol[0:5] == "sigma":
            any_sigma = s
            sigma_chk = True
    c2p_chk = False
    if largest_Cn:
        for s in symels:
            if s.symbol[0] == "C" and int(rgx.search(s.symbol).groups()[0]) == 2:
                if not issame_axis(s.vector, largest_Cn.vector):
                    c2p_chk = True
                    c2p = s.vector
        paxis = largest_Cn.vector
        if c2p_chk:
            saxis = c2p
        elif any_sigma:
            saxis = np.cross(any_sigma.vector, paxis)
        else:
            pass
    elif sigma_chk:
        paxis = any_sigma.vector
    else:
        pass
    
    return paxis, saxis
