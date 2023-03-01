import numpy as np
import psi4
import json
from dataclasses import dataclass
from copy import deepcopy
tol = 1e-4
@dataclass
class Atom():
    Z:int
    mass:float
    xyz:np.array

@dataclass
class SEA():
    label:str
    subset:np.array
    axis:np.array

class Molecule():
    def __init__(self, atoms, coords, masses) -> None:
        self.atoms = np.asarray(atoms)
        try:
            self.natoms = len(self.atoms)
        except TypeError:
            self.natoms = 1
        self.coords = np.asarray(coords)
        self.masses = np.asarray(masses)

    @classmethod
    def from_schema(cls, squeema):
        atoms = squeema["elem"]
        natoms = len(atoms)
        coords = np.reshape(squeema["geom"], (natoms,3))
        masses = squeema["mass"]
        return cls(atoms, coords, masses)

    def __getitem__(self, i):
        return Molecule(self.atoms[i], self.coords[i,:], self.masses[i])

    def __len__(self):
        return self.natoms

    def __eq__(self, other):
        if isinstance(other, Molecule):
            return (other.atoms == self.atoms).all() and (other.masses == self.masses).all() and np.allclose(other.coords, self.coords, atol=tol)

    def find_com(self):
        com = np.zeros(3)
        for i in range(self.natoms):
            com += self.masses[i]*self.coords[i,:]
        return com / sum(self.masses)

    def translate(self, r):
        for i in range(self.natoms):
            self.coords[i,:] -= r
        
    def is_at_com(self):
        if sum(abs(self.find_com())) < tol:
            return True
        else:
            return False

    def _transform(self, M):
        self.coords = np.dot(self.coords,np.transpose(M))

    def transform(self, M):
        new_mol = deepcopy(self)
        new_mol.coords = np.dot(new_mol.coords,np.transpose(M))
        return new_mol

    def distance_matrix(self):
        dm = np.zeros((self.natoms,self.natoms))
        for i in range(self.natoms):
            for j in range(i,self.natoms):
                dm[i,j] = np.sqrt(sum((self.coords[i,:]-self.coords[j,:])**2))
                dm[j,i] = dm[i,j]
        return dm

    def find_SEAs(self):
        dm = self.distance_matrix()
        out = []
        nate_thang = []
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                a_idx = np.argsort(dm[i,:])
                b_idx = np.argsort(dm[j,:])
                z = dm[i,a_idx] - dm[j,b_idx]
                chk = True
                for k in z:
                    if abs(k) < tol:
                        continue
                    else:
                        chk = False
                if chk:
                    out.append((i,j))
        nonos = []
        SEAs = []
        for i in range(self.natoms):
            if i in nonos:
                continue
            else:
                biggun = [i]
            
            for k in out:
                if i in k:
                    if i == k[0]:
                        biggun.append(k[1])
                        nonos.append(k[1])
                    else:
                        biggun.append(k[0])
                        nonos.append(k[0])
            SEAs.append(SEA("", biggun, np.zeros(3)))
        return SEAs

def rotation_matrix(axis, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # NOT NORMALIZING AXIS!!!
    M = np.zeros((3,3))
    M += 1 - cos_t
    for i in range(3):
        for j in range(3):
            M[i,j] *= axis[i]*axis[j]
    M += np.asarray([[cos_t, -axis[2]*sin_t, axis[1]*sin_t],[axis[2]*sin_t, cos_t, -axis[0]*sin_t],[-axis[1]*sin_t, axis[0]*sin_t, cos_t]])
    return M

def reflection_matrix(axis):
    M = np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            if i == j:
                M[i,i] = 1 - 2*(axis[i]**2)
            else:
                M[i,j] = -2 * axis[i] * axis[j]
                M[j,i] = M[i,j]
    return M

def inversion_matrix():
    return -1*np.identity(3)

def Cn(axis, n):
    theta = 2*np.pi/n
    return rotation_matrix(axis, theta)

def Sn(axis, n):
    return np.dot(Cn(axis, n), reflection_matrix(axis))

def isequivalent(A,B):
    h = []
    for i in range(A.natoms):
        for j in range(B.natoms):
            if A.masses[i] == B.masses[j]:
                zs = abs(A.coords[i,:]-B.coords[j,:])
                if np.allclose(zs, [0,0,0], atol=tol):
                    h.append(j)
                    break
    if len(h) == A.natoms:
        return True
    return False

def calcmoit(atoms):
    I = np.zeros((3,3))
    atoms.translate(atoms.find_com())
    for i in range(3):
        for j in range(3):
            if i == j:
                for k in range(atoms.natoms):
                    I[i,i] += atoms.masses[k]*(atoms.coords[k,(i+1)%3]**2+atoms.coords[k,(i+2)%3]**2)
            else:
                for k in range(atoms.natoms):
                    I[i,j] -= atoms.masses[k]*atoms.coords[k,i]*atoms.coords[k,j]
    return I

def eat_shit(a):
    pass

if __name__ == "__main__":
    with open("xyz/water.xyz", "r") as fn:
        strang = fn.read()
    psimol = psi4.core.Molecule.from_string(strang)
    beebus = psimol.to_schema("psi4")
    mol = Molecule.from_schema(beebus)
    #print(mol.coords)
    #print(mol.find_com())
    seas = mol.find_SEAs()
    print(seas)