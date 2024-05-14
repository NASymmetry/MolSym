"""
This code generates the rotation matrices for real spherical harmonics by a recursion relation
found in "Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion."
J. Ivanic and K. Rudenberg: doi/10.1021/jp953350u
"""

import numpy as np
from molsym.salcs.function_set import FunctionSet

#0, 1+, 1-, 2+, 2- .... l+, l-
def generateshuffle(l):
    count = 0
    beeb = [x for x in range(0, l*2 + 1)]
    for ind in range(0, l + 1):
        squeeb = beeb[ind - count]
        beeb = np.delete(beeb, ind - count)
        beeb = np.insert(beeb, 2 * l - ind - count, squeeb)
        count += 1
    beeb = list(beeb)
    return beeb

def generateRotations(Lmax, rot):
    Rsh = []
    rrot = adapt(rot)
    Rsh.append(np.eye(1))
    l = 1
    while l < Lmax + 1:
        if l == 1:
            psi4 = True
            #psi4 = False
            if psi4:
                Rsh.append(rrot)
            else:
                Rsh.append(rrot)
        if l > 1:
            R = np.zeros((2*l + 1, 2*l + 1))
            for m1 in range(-l,l + 1):
                for m2 in range(-l,l + 1):
                    u, v, w = UWVCoefficient(l, m1, m2)
                    if u != 0:
                        u *= Ufun(l, m1, m2, rrot, Rsh)
                    if v != 0:
                        v *= Vfun(l, m1, m2, rrot, Rsh)
                    if w != 0:
                        w *= Wfun(l, m1, m2, rrot, Rsh)
                    R[m1 + l, m2 + l] = u + v + w
            Rsh.append(R)
        l += 1
   
    if Lmax >= 1:
        Rsh[1] = rot[:, [2, 0, 1]] 
        Rsh[1] = Rsh[1][[2, 0, 1], :] 
    for r, rsh in enumerate(Rsh):
        if r > 1:
            beeb = generateshuffle(r)
            Rsh[r] = Rsh[r][:, beeb]
            Rsh[r] = Rsh[r][beeb, :]
    return Rsh

#generates u, w, and v coefficients, eq. 8.1, found in Table 1 of reference
def UWVCoefficient(l, m1, m2):
    delta = bool(0 == m1)
    if abs(m2) < l:
        denom = (l + m2)*(l - m2)
    else:
        denom = (2.0*l)*(2.0*l - 1.0)
    u = (((l + m1)*(l - m1)) / denom)**(1/2)
    v = 0.5*(((1 + delta)*(l + abs(m1) -1)*(l + abs(m1)) / denom) ** (1/2))*(1 - 2*delta)
    w = -0.5*(((l - abs(m1) - 1)*(l - abs(m1)) / denom) ** (1/2))*(1 - delta)
    return u, v, w

#generates function U, eq. 8.1, found in Table 2 of reference
def Ufun(l, m1, m2, rot, Rsh):
    return Pfun(l, 0, m1, m2, rot, Rsh)

#generates function V, eq. 8.1, found in Table 2 of reference, with a sign correction (DERIVE)
def Vfun(l, m1, m2, rot, Rsh):
    if m1 == 0:
        V = Pfun(l, 1, 1, m2, rot, Rsh) + Pfun(l, -1, -1, m2, rot, Rsh)
    elif m1 == 1:
        V = np.sqrt(2)*Pfun(l,  1, 0, m2, rot, Rsh)
    elif m1 == -1:
        V = np.sqrt(2)*Pfun(l, -1, 0, m2, rot, Rsh)
    elif m1 > 0:
        V = Pfun(l, 1, m1 -1, m2, rot, Rsh) - Pfun(l, -1, -m1 + 1, m2, rot, Rsh) ##SIGN CORRECTION
    else:
        V = Pfun(l, 1, m1 +1, m2, rot, Rsh) + Pfun(l, -1, -m1 - 1, m2, rot, Rsh)
    return V

#generates function W, eq. 8.1, found in Table 2 of reference
def Wfun(l, m1, m2, rot, Rsh):
    if m1 > 0:
        W = Pfun(l, 1, m1 +1, m2, rot, Rsh) + Pfun(l, -1, -m1 -1, m2, rot, Rsh)
    elif m1 < 0:
        W = Pfun(l, 1, m1 -1, m2, rot, Rsh) - Pfun(l, -1, -m1 +1, m2, rot, Rsh)
    return W

#generates function P, eq. 8.1, found in Table 2 of reference
def Pfun(l, i, m1, m2, rot, Rsh):
    rsh = Rsh[l - 1]
    dl = len(Rsh[l - 1])
    ol = int((dl -1) / 2)
    if m2 == l:
        P1 = rot[i + 1, 2] * rsh[m1 + ol, l - 1 + ol]
        P2 = rot[i + 1, 0] * rsh[m1 + ol, 1 - l + ol]
        P = P1 - P2
    elif m2 == -l:
        P1 = rot[i + 1, 2] * rsh[m1 + ol, 1 - l + ol]
        P2 = rot[i + 1, 0] * rsh[m1 + ol, l - 1 + ol]
        P = P1 + P2
    else: 
        P = rot[i + 1, 1] * rsh[m1 + ol, m2 + ol]
    return P
#For l = 1, RSH and cartesians are the same, just rotate for convention
#y -> x
#z -> y
#x -> z
def adapt(rot):
    rrot = np.zeros((3,3))
    rrot[0,0] = rot[1,1]
    rrot[0,1] = rot[1,2]
    rrot[0,2] = rot[1,0]
    rrot[1,0] = rot[2,1]
    rrot[1,1] = rot[2,2]
    rrot[1,2] = rot[2,0]
    rrot[2,0] = rot[0,1]
    rrot[2,1] = rot[0,2]
    rrot[2,2] = rot[0,0]
    return rrot

#count the number of shells between an atom
def obstruct(atom1, atom2, nbas_vec):
    if atom1 == atom2:
        obstruction = 0
    else:
        obstruction = 0
    
        A = int(min(atom1,atom2) + 1)
        B = int(max(atom1,atom2))
        for x in nbas_vec[A:B]:
            obstruction += x
    return obstruction

#collect parameterized rsh rotations for each symmetry operation
def rotate_em(maxam, ops):
   rsh_rot_per_op = []
   for i, op in enumerate(ops):
       rsh_rot_per_op.append(generateRotations(maxam, op.rrep))
   return rsh_rot_per_op

class SphericalHarmonics(FunctionSet):
    def __init__(self, symtext, fxn_list) -> None:
        self.fxns = fxn_list
        self.symtext = symtext
        self.maxam = self.get_maxam()
        self.nbas_vec = []
        for i in self.fxns:
            self.nbas_vec.append(sum([2*l+1 for l in i]))
        
        self.big_info = [] # basis_function: atom idx, shell_idx, l, ml (not actual ml)
        for atom_idx in range(len(self.symtext.mol)):
            for shell_idx in range(len(self.fxns[atom_idx])):
                l = self.fxns[atom_idx][shell_idx]
                for bfxn in range(2*l+1): # ml = 0, 1, -1, 2, -2, ...
                    self.big_info.append([atom_idx, shell_idx, l, bfxn])
        
        self.fxn_map = self.get_fxn_map()
        self.SE_fxns = self.get_symmetry_equiv_functions()
    def __len__(self):
        return sum(self.nbas_vec)

    def get_symmetry_equiv_functions(self):
        symm_equiv = []
        done = []
        for bfxn_i in range(len(self)):
            #print("Done: ",done)
            if bfxn_i in done:
                continue
            equiv_set = []
            for sidx in range(len(self.symtext)):
                v = self.fxn_map[bfxn_i, sidx, :]
                #print(v)
                lv = np.isclose(v, 0.0, atol=1e-12)
                equiv_set += [i for i in range(len(lv)) if not lv[i]]
            #print(equiv_set)
            reduced_equiv_set = list(set(equiv_set))
            symm_equiv.append(reduced_equiv_set)
            done += reduced_equiv_set

        return symm_equiv

    def get_fxn_map(self):
        # Spherical harmonic map for l up to maxam, not including l = 0
        self.rotated = rotate_em(self.maxam, self.symtext.symels) # symel x l x ml
        fxn_map = np.zeros((len(self), len(self.symtext), len(self))) # basis_function x symel x basis_function
        for bfxn_i in range(len(self)):
            atom_i, sh, l, ml = self.big_info[bfxn_i]
            for sidx in range(len(self.symtext)):
                rotate_result = self.rotated[sidx]
                atom_j = self.symtext.atom_map[atom_i,sidx]
                if l == 0:
                    result = np.eye(1)
                else:
                    result = rotate_result[l][:, ml]
                result = np.squeeze(result)
                if atom_j == atom_i:
                    fxn_map[bfxn_i, sidx, bfxn_i-ml:bfxn_i-ml+(2*l)+1] += result
                else:
                    obstruction = obstruct(atom_i, atom_j, self.nbas_vec)
                    offset = obstruction + self.nbas_vec[atom_i]
                    if atom_i > atom_j:
                        offset *= -1
                    fxn_map[bfxn_i, sidx, bfxn_i-ml+offset:bfxn_i-ml+offset+(2*l)+1] += result
        return fxn_map

    def get_maxam(self):
        maxam = 0
        for i in self.fxns:
            maxam = max(maxam, max(i, default=0))
        return maxam

    def special_function(self, salc, coord, sidx, irrmat):
        dim = irrmat[sidx,:,:].shape[0]
        new_vec = self.fxn_map[coord, sidx, :]
        #salc[:,:,:] += np.multiply((irrmat[sidx, :, :]).reshape(dim**2,1), new_vec).reshape((dim, dim, new_vec.size))
        for i in range(dim):
            for j in range(dim):
                salc[i,j,:] += irrmat[sidx,i,j] * new_vec
        return salc

