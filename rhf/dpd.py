import numpy as np
import time
from itertools import combinations_with_replacement
#from dptable import DPTable

"""
Python Class for handling symmetry in tensor operations

no idea what this will look like...

Should handle BDMatrix class

Maybe add some attributes to BDMatrix so PyDPD can do lookups and sorts

1st project is to symmetry adapt MP2 just like psi4
"""

class DPD():

    def __init__(self, orb_idx, symtext):
    #def __init__(self, tensor, orb_idx, symtext):
        #print("nothing to init just yet")
        #self.tensor = tensor
        self.orb_idx = orb_idx
        self.symtext = symtext
        #maybe take MolSym symtext object

    #string could be something that would be used in np.einsums
    #or lists corresponding to indices
    def define(self, string):
        #controls how tensors are ultimately rearranged
        return TSIR_blocks
    
    def dp_contains(self, irrep, a, b, *args):
        #print("inside dp contains")
        ctab = self.symtext.chartable
        #print(vars(ctab))
        a = ctab.characters[a]
        b = ctab.characters[b]
        #print(f"the irrep {irrep}")
        #print(self.symtext.chartable)
        #print(self.symtext.chartable.class_orders)
        chars = a * b
        for arg in args:
            #print(f"arg {arg}")
            chars *= ctab.characters[arg]
        s = sum(chars * ctab.class_orders * irrep)
        n = s // sum(ctab.class_orders)
        if n > 0:
            return True
        return False
    
    def dp_contains_tsir(self, a, b, *args):
        ctab = self.symtext.chartable
        a = ctab.characters[a]
        b = ctab.characters[b]
        #print(f"characters a and b {a, b}")
        chars = a * b
        for arg in args:
            chars *= ctab.characters[arg]
        #print(f"chars {chars}")
        #print(f"chars {chars * ctab.class_orders}")
        s = sum(chars * ctab.class_orders * ctab.characters[0])
        #print(s)
        n = s / sum(ctab.class_orders)
        #print(f"n {n}")
        if np.isclose(n, 0, atol = 1e-4):
            #print("this should be 1")
            return False
        return True
        #if n > 0:
        #    return True
        #return False

    #makes sure mu * nu = tsir && rho * sigma == tsir of symm adapted tei,  for fock build
    def lookup_hf_ERI_J(self, tensor):
        self.tensor = tensor
        before = time.time()
        #print("special case lookup")
        self.nonzero_blocks = []
        self.i_symmetry = []
        self.k_symmetry = []
        for i in range(0, len(self.orb_idx)):
            if len(self.orb_idx[i]) != 0:
                for j in range(0, len(self.orb_idx)):
                    if len(self.orb_idx[j]) != 0:
                        if self.dp_contains_tsir(i, j):
                            for k in range(0, len(self.orb_idx)):
                                if len(self.orb_idx[k]) != 0:
                                    for l in range(0, len(self.orb_idx)):
                                        if len(self.orb_idx[l]) != 0:
                                            if self.dp_contains_tsir(k, l):
                                                #print(f" ijkl {i, j, k, l}")
                                                self.i_symmetry.append(i)
                                                self.k_symmetry.append(k)
                                                self.nonzero_blocks.append([i, j, k, l])
        self.twod_tensor = self.indices()
        #print(self.twod_tensor)
        #print(beebus)
    
    #makes sure mu * nu = tsir && rho * sigma == tsir of symm adapted tei,  for fock build
    def lookup_hf_ERI_K(self, tensor):
        self.tensor = tensor
        before = time.time()
        #print("special case lookup")
        self.nonzero_blocks = []
        for i in range(0, len(self.orb_idx)):
            if len(self.orb_idx[i]) != 0:
                for j in range(0, len(self.orb_idx)):
                    if len(self.orb_idx[j]) != 0:
                        if self.dp_contains_tsir(i, j):
                            for k in range(0, len(self.orb_idx)):
                                if len(self.orb_idx[k]) != 0:
                                    for l in range(0, len(self.orb_idx)):
                                        if len(self.orb_idx[l]) != 0:
                                            if self.dp_contains_tsir(k, l):
                                                #print(f" ijkl {i, j, k, l}")
                                                self.nonzero_blocks.append([i, k, j, l])
        self.twod_tensor = self.indices()
    
    def indices(self):
        #print("inside indices")
        twod_tensor = []
        tot = 0
        for block in self.nonzero_blocks:
            i_idx, j_idx, k_idx, l_idx = self.orb_idx[block[0]], self.orb_idx[block[1]], self.orb_idx[block[2]], self.orb_idx[block[3]]
            twod_tensor_b = np.zeros((len(i_idx) * len(j_idx), len(k_idx) * len(l_idx)))
            for i, ib in enumerate(i_idx): 
                for j, jb in enumerate(j_idx): 
                   for k, kb in enumerate(k_idx): 
                       for l, lb in enumerate(l_idx):
                           ij = len(j_idx) * i + j
                           kl = len(l_idx) * k + l
                           #print(f"ij kl {ij, kl}")
                           #print(f"ijkl {i, j, k, l}")
                           #print(f"ijkl {ib, jb, kb, lb}")
                           #print(self.tensor[ib, jb, kb, lb])
                           twod_tensor_b[ij,kl] = self.tensor[ib,jb,kb,lb]
            twod_tensor.append(twod_tensor_b)
        return twod_tensor

