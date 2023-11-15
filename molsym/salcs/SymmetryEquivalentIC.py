import copy
from dataclasses import dataclass
import numpy as np
from numpy import linalg as LA
import warnings

@dataclass
class IC():
    AtomIndices:np.array #length-N array of atom indices for the N-centered int coord
    Stabilizer:np.array #array of operations that maps the funciton onto itself, can be length 0

@dataclass
class SEIC():
    ICindices:np.array #length-N array of ic indices for N-ICs that are symmetry equivalent

class InternalCoordinates():
    def __init__(self, int_coords, mol, ic_types):
        warnings.warn("Using deprecated code")
        self.ic_list = int_coords
        self.mol = mol
        self.ic_types = ic_types
     
    def stable(self, i):
        return np.where(self.ic_map[i,:]==i)[0] 
    
    def operate_on_ic(self, ic, ic_index,  symtext, op):
        #print("Inside operate on IC, s vec first")
        xyzop = symtext.symels[op].rrep
        symbol = symtext.symels[op].symbol
        ic_type = self.ic_types[ic_index][0]
        if symbol[0] == "i" or symbol[0] == "s" or symbol[0] == "S":
            #print(symbol)
            if ic_type == "D" or ic_type == "O": 
                self.phase_map[ic_index, op] = -1.0
            elif ic_type == "L":
                self.phase_map[ic_index, op] = -1.0
        mapped_ic = []
        for a, atom in enumerate(ic):
            atom2 = int(symtext.atom_map[atom,op])
            mapped_ic.append(atom2)
            #what is the phase?
        index, phase = self.ic_index(mapped_ic) 
        return mapped_ic, index, phase 
    
    def run(self, SEAs, symtext):
         
        self.IC_map(SEAs, symtext)
    #create a map data structure for each coordinate. 
    #where does it map to, what is it's index
    #what is its phase?
    def IC_map(self, SEAs, symtext):
        # Returns Symel by IC array
        # loop over the internal coordinates
        self.ICs = []
        self.SEICs = []
        seics = []
        self.ic_map = np.zeros((len(self.ic_list), len(symtext)), dtype=np.int32)
        self.phase_map = np.zeros((len(self.ic_list), len(symtext)))
        S = (len(self.ic_list), len(symtext))
        self.phase_map = np.ones(S)
        #loop over the internal coordinate list
        for i, ic in enumerate(self.ic_list):
            seic_set = []
            if ic not in seic_set:
                seic_set.append(i)
            for sidx, symel in enumerate(symtext.symels):
                #loop over the atom indices in the list
                #mapped ic is the ic atom indices list, index is the the mapped_ic index
                mapped_ic, index, phase = self.operate_on_ic(ic, i, symtext, sidx)
                #mapped_ic, index, phase = self.operate_on_ic(ic, i, symtext, int(Class))
                #print(f"mapped_ic {mapped_ic}")
                self.ic_map[i, sidx] = index
                #self.ic_map[i, int(Class)] = index
                #trial
                self.phase_map[i, sidx] *= phase
                if index not in seic_set:
                    seic_set.append(index)
            #print("do we have seics?")
            #print(seics)
            flat_list = np.array([item for sublist in seics for item in sublist])
            
            intersection = np.setdiff1d(np.array(seic_set), flat_list)
            
            if len(intersection) > 0:
                seics.append(seic_set)
                self.SEICs.append(SEIC(np.array(seic_set)))
            stabby = self.stable(i)
            self.ICs.append(IC(ic, stabby))
        
    def ic_index(self, ic):
        #print("inside_ic_index")
        #print(ic)
        if len(ic) > 3:
            ic2 = copy.deepcopy(ic)
            #print(f"ic {ic}")
            ic2[2], ic2[3] = ic[3], ic[2]
            #print(self.ic_list)
            for f, funky in enumerate(self.ic_list):
                #print(f"funky {funky}")
                if ic == funky:
                    #print("condition 1")
                    return f, 1
                elif list(reversed(ic)) == funky:
                    #print("condition 2")
                    return f, 1
                elif ic2 == funky:
                    #print("condition 3")
                    return f, -1
                elif list(reversed(ic2)) == funky:
                    #print("condition 4")
                    return f, -1
            

        else:
            #print("is this the f up?")
            for f, funky in enumerate(self.ic_list):
                #print(f"f {f} funky {funky}")
                if ic == funky:
                    return f, 1
                elif list(reversed(ic)) == funky:
                    return f, 1
