import copy
from copy import deepcopy
import numpy as np
from .function_set import FunctionSet

class InternalCoordinates(FunctionSet):
    def __init__(self, fxn_list, symtext) -> None:
        self.ic_list = [i[0] for i in fxn_list]
        self.ic_types = [i[1] for i in fxn_list]
        super().__init__(fxn_list, symtext)
#[([14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 28, 29], [[14, 26], [15, 17], [18, 20], [21, 23], [24, 28], [27, 29]])]
    def operate_on_ic(self, ic_idx, symop):
        symbol = self.symtext.symels[symop].symbol
        ic_type = self.ic_types[ic_idx][0]
        #self.phase_map = np.ones((len(self.ic_list), len(self.ic_list)))
        if symbol[0] == "i" or symbol[0] == "S" or symbol[0] == "s": # s is for sigma
            if ic_type == "D" or ic_type == "O":
                self.phase_map[ic_idx, symop] = -1.0
                print(self.phase_map[ic_idx, symop])
            elif ic_type == "L":
                self.phase_map[ic_idx, symop] = -1.0
                print(self.phase_map[ic_idx, symop])
           
            ## Are we doing this twice??? TODO here
            #if ic_type in ["D", "O", "L"]: 
            #    self.phase_map[ic_idx, symop] = -1.0
            ##if ic_type == "D" or ic_type == "O": 
            ##    self.phase_map[ic_idx, symop] = -1.0
            ##elif ic_type == "L":
            ##    self.phase_map[ic_idx, symop] = -1.0
        mapped_ic = []
        print(f"the ic {self.ic_list[ic_idx]}")
        for atom in self.ic_list[ic_idx]:
            atom2 = int(self.symtext.atom_map[atom, symop])
            mapped_ic.append(atom2)
        index, phase = self.ic_index(mapped_ic)
        #print(f"this is the phase {phase}")
        return index, phase
    def first_or_last(self, a, b):
        return a[:2] == b[:2] or a[1:] == b[1:] 
    
    def is_coplanar(self, a, b):
        #print(f"coordinates {a} {b}") 
        #print("is cloplanar?")
        #print(self.symtext.mol)
        #print(vars(self.symtext.mol))
        diff = np.setdiff1d(a, b) #return unique values of a that are not in b. If length = 1, angles share two common indices
        #print(f"the diff {diff}")
        if len(diff) == 1:
            #1st, check if they are symmetry equivalent
            #print("the reverse")
            #print(list(reversed(b)))
            #print("try again")
            #print(b[::-1])
            #2nd, check if first two elements or last two elements are the same
            if self.first_or_last(a, b):
                #print(f"check if {a} {b} are coplanar")
                unique = np.unique(a + b)
                candidate = self.symtext.mol.coords[unique]
                centroid = np.mean(candidate, axis = 0)
                _, magnitudes, axes = np.linalg.svd(candidate - centroid, full_matrices = False)
                planarity_tol = 1e-6
                return magnitudes[2] < planarity_tol
            elif self.first_or_last(a, list(reversed(b))):
                #print(f"check if {a} {b} are coplanar")
                unique = np.unique(a + b)
                candidate = self.symtext.mol.coords[unique]
                centroid = np.mean(candidate, axis = 0)
                _, magnitudes, axes = np.linalg.svd(candidate - centroid, full_matrices = False)
                planarity_tol = 1e-6
                return magnitudes[2] < planarity_tol
            else:
                #print(f"{a} {b} not a pair")
                return False
                 
            #2nd, check if they are in the same  
            #print(f"diff {diff}") 
            unique = np.unique(a + b)
            #print(f"unique {unique}")
            candidate = self.symtext.mol.coords[unique]
            #print(f"candidate {candidate}")
            centroid = np.mean(candidate, axis = 0)
            #print(f"centroid {centroid}")
            _, magnitudes, axes = np.linalg.svd(candidate - centroid, full_matrices = False)
            #axes: 3 unitary axis of ellipsoids, magnitudes: their magnitudes, sorted
            planarity_tol = 1e-6
            return magnitudes[2] < planarity_tol
 
    def check_coplanar(self, SEICs):
        sets = []
        for function_set in SEICs:
            coplanar_pairs = []
            #print(f"function set {function_set}")
            #print(function_set[0])
            good = False
            if self.ic_types[function_set[0]][0] == 'A':
                print(f"function set {function_set}")
                #sets.append(function_set)
                for i, ici in enumerate(function_set):
                    for j, icj in enumerate(function_set):
                        if i != j and j > i:
                            #print(f"ici and icj {ici, icj}")
                            if self.is_coplanar(self.ic_list[ici], self.ic_list[icj]):
                                #print(f"This is a pair!")
                                print(f"{self.ic_list[ici]} {self.ic_list[icj]}")
                                coplanar_pairs.append([ici, icj])
                                good = True
                            else:
                                continue
                if good:
                    sets.append((function_set, coplanar_pairs))
                for f in function_set:
                    print(self.fxn_map[f,:]) 
        print("These are the coplanar pairs")
        print(sets)
        #print(coplanar_pairs)
        #print(self.fxn_map)
        #the first of each pair is the reference
        for seti in sets:
            for pair in seti[1]:
                for s, fset in enumerate(seti[0]): 
                    print("pair and fset")
                    print(pair)
                    print(fset) 
                    for idx, a in enumerate(self.fxn_map[fset, :]):
                        if pair[1] == a:
                            print(f"idx {idx}")
                            print(self.fxn_map[fset, :])
                            self.fxn_map[fset, idx] = pair[0]
                            #self.phase_map[fset, idx] *= -1 #self.phase_map[pair[0], idx] * -1 
                            print(self.fxn_map[fset, :])
        #print(beebus)
        #            
        #            print(self.fxn_map[pair[0], :])
        #            for idx, a in enumerate(self.fxn_map[pair[0], :]):
        #                if pair[1] == a:
        #                    print(f"idx {idx}")
        #                    #self.fxn_map[pair[0], idx] = pair[0]
        #                    #self.phase_map[pair[0], idx] *= -1 #self.phase_map[pair[0], idx] * -1 
        #                    print(self.fxn_map[pair[0], :])
            #if pair[1] in self.fxn_map[pair[0], :]:
            #    idx = list(self.fxn_map[pair[0], :]).index(pair[1])
            #    print(f"the idx {idx}")
             
        #self.method_reverse_lookup() 
        #for pair in coplanar_pairs:
        #print(beebus)
        
        #for i, ici in enumerate(self.ic_list):
        #    for j, icj in enumerate(self.ic_list):
        #        if i != j and self.ic_types[i][0] == 'A' and self.ic_types[j][0] == 'A':
        #            print(f"Its an angle {i} {self.ic_types[i]} {ici}")
        #            print(f"Its an angle {j} {self.ic_types[j]} {icj}")
        #            if self.is_coplanar(ici, icj):
        #                print(f"This is a pair!")
    def method_reverse_lookup(self):
        count = 0
        for i, x in enumerate(self.fxn_map):
            count += 1
        print(count)
        print(beebus)
#[([14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 28, 29], [[14, 26], [15, 17], [18, 20], [21, 23], [24, 28], [27, 29]])]
    def get_fxn_map(self):
        ic_map = np.zeros((len(self.ic_list), len(self.symtext)), dtype=np.int32)
        S = (len(self.ic_list), len(self.symtext))
        self.phase_map = np.ones((len(self.ic_list), len(self.ic_list)))
        #phase_map = np.ones(S)
        for ic_idx in range(len(self)):
            for sidx, symel in enumerate(self.symtext.symels):
                index, phase = self.operate_on_ic(ic_idx, sidx)
                #print(f"this is the index! {index}")
                #if index == 26:
                #    #index = 14
                #    phase *= -1 
                #if index == 17:
                #    #index = 15 
                #    phase *= -1 
                #if index == 20:
                #    #index = 18 
                #    phase *= -1 
                #if index == 23:
                #    #index = 21 
                #    phase *= -1 
                #if index == 28:
                #    #index = 24 
                #    phase *= -1 
                #if index == 29:
                #    #index = 27 
                #    phase *= -1 
                ic_map[ic_idx, sidx] = index
                self.phase_map[ic_idx, sidx] *= phase
        print(self.phase_map)
        #print(beebus) 
        return ic_map, self.phase_map

    def get_symmetry_equiv_functions(self):
        SEICs = []
        done = []
        for ic_idx in range(len(self.ic_list)):
            if ic_idx in done:
                continue
            else:
                seics = []
                for symel_idx in range(len(self.symtext)):
                    seics.append(self.fxn_map[ic_idx, symel_idx])
                reduced_seics = list(set(seics))
                done += reduced_seics
                SEICs.append(reduced_seics)
        
        #self.check_coplanar(SEICs)
        return SEICs

    def ic_index(self, ic):
        print(f"This is the ic {ic}")
        if len(ic) > 3:
            ic2 = deepcopy(ic)
            ic2[2], ic2[3] = ic[3], ic[2]
            for f, funky in enumerate(self.ic_list):
                # TODO here
                if ic == funky:
                    return f, 1
                elif list(reversed(ic)) == funky:
                    return f, 1
                elif ic2 == funky:
                    return f, -1
                elif list(reversed(ic2)) == funky:
                    return f, -1
        elif len(ic) == 3:
            print("this be and angle")
            #first, loop through IC list, only after the list is exhausted, loop through reverse indices
            for f, funky in enumerate(self.ic_list):
                if ic == funky:
                    print("mapped on")
                    return f, 1
                elif list(reversed(ic)) == funky:
                    print("reversed")
                    return f, 1
            print(f"ic {ic} didn't meet the criteria")
            ic2 = copy.deepcopy(ic)
            #ic2[0], ic2[2] = ic[2], ic[0]
            for f, funky in enumerate(self.ic_list):
                if len(funky) == 3:
                    #print(funky)
                    if ic[0] == funky[0] and ic[1] == funky[1]:
                        print("Oboi")
                        print(funky)
                        return f, -1

        else:
            for f, funky in enumerate(self.ic_list):
                if ic == funky:
                    return f, 1
                elif list(reversed(ic)) == funky:
                    return f, 1

    def span(self, SE_fxn):
        chorker_loaf = np.zeros(self.fxn_map.shape)
        for i in range(self.fxn_map.shape[0]):
            for j in range(self.fxn_map.shape[1]):
                if self.fxn_map[i,j] == i:
                    chorker_loaf[i,j] = 1
        rhorker_loaf = chorker_loaf[SE_fxn, :]
        shorker_loaf = np.sum(rhorker_loaf, axis=0)
        rshorker_loaf = np.zeros(self.symtext.chartable.characters.shape[1])
        for i in range(len(self.symtext)):
            rshorker_loaf[self.symtext.class_map[i]] = shorker_loaf[i]
        span = np.zeros(len(self.symtext.chartable.irreps), dtype=np.int32)
        for idx, irrep in enumerate(self.symtext.chartable.irreps):
            n = round(np.sum(rshorker_loaf * self.symtext.chartable.class_orders * self.symtext.chartable.characters[idx,:]) / self.symtext.order)
            span[idx] = n
        return span
