import numpy as np
#import psi4
import qcelemental as qcel
import json
from dataclasses import dataclass
from copy import deepcopy
global_tol = 1e-6
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
        self.tol = 1e-5
        self.atoms = np.asarray(atoms)
        try:
            self.natoms = len(self.atoms)
        except TypeError:
            self.natoms = 1
        self.coords = np.asarray(coords)
        self.masses = np.asarray(masses)

    @classmethod
    def from_schema(cls, schema):
        atoms = schema["symbols"] # was symbols
        natoms = len(atoms)
        coords = np.reshape(schema["geometry"], (natoms,3)) # was geometry
        # As of now, QCElemental seems to have issues assigning masses, so I do it
        masses = np.zeros(natoms)
        for (idx, symb) in enumerate(atoms):
            masses[idx] = qcel.periodictable.to_mass(symb)
        #masses = schema["masses"]
        return cls(atoms, coords, masses)

    @classmethod
    def from_file(cls, fn):
        with open(fn, "r") as lfn:
            strang = lfn.read()
        schema = qcel.models.Molecule.from_data(strang).dict()
        return cls.from_schema(schema)

    @classmethod
    def from_psi4_schema(cls, schema):
        atoms = schema["elem"] # was symbols
        natoms = len(atoms)
        coords = np.reshape(schema["geom"], (natoms,3)) # was geometry
        # As of now, QCElemental seems to have issues assigning masses, so I do it
        masses = np.zeros(natoms)
        for (idx, symb) in enumerate(atoms):
            masses[idx] = qcel.periodictable.to_mass(symb)
        #masses = schema["masses"]
        return cls(atoms, coords, masses)

    def __repr__(self) -> str:
        rstr = "MolSym Molecule:\n"
        for i in range(self.natoms):
            rstr += f"   {self.atoms[i]:3s}   {self.coords[i,0]:12.8f}   {self.coords[i,1]:12.8f}   {self.coords[i,2]:12.8f}\n"
        return rstr

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, i):
        return Molecule(self.atoms[i], self.coords[i,:], self.masses[i])

    def __len__(self):
        return self.natoms

    def __eq__(self, other):
        # Select higher tolerance
        if self.tol >= other.tol:
            eq_tol = self.tol
        else:
            eq_tol = other.tol
        if isinstance(other, Molecule):
            return (other.atoms == self.atoms).all() and (other.masses == self.masses).all() and np.allclose(other.coords, self.coords, atol=eq_tol)

    def find_com(self):
        com = np.zeros(3)
        for i in range(self.natoms):
            com += self.masses[i]*self.coords[i,:]
        return com / sum(self.masses)

    def translate(self, r):
        for i in range(self.natoms):
            self.coords[i,:] -= r
        
    def is_at_com(self):
        if sum(abs(self.find_com())) < self.tol:
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
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                a_idx = np.argsort(dm[i,:])
                b_idx = np.argsort(dm[j,:])
                z = dm[i,a_idx] - dm[j,b_idx]
                chk = True
                for k in z:
                    if abs(k) < self.tol:
                        continue
                    else:
                        chk = False
                if chk:
                    out.append((i,j))
        skip = []
        SEAs = []
        for i in range(self.natoms):
            if i in skip:
                continue
            else:
                biggun = [i]
            
            for k in out:
                if i in k:
                    if i == k[0]:
                        biggun.append(k[1])
                        skip.append(k[1])
                    else:
                        biggun.append(k[0])
                        skip.append(k[0])
            SEAs.append(SEA("", biggun, np.zeros(3)))
        return SEAs

    def symmetrize(self, asym_tol=0.05):
        # This code might be bad. Consider removing. Interatomic distances ---> Cart. not always well defined
        print("Warning! Using this symmetrize (the one in Molecule) may fail!")
        dm = self.distance_matrix()
        SEAs = self.find_SEAs()
        new_dm = deepcopy(dm)
        # Symmetrize interatomic distance matrix
        for sea in SEAs:
            if len(sea.subset) < 2:
                continue
            for dm_el_0 in range(self.natoms):
                matches = [(sea.subset[0], dm_el_0)]
                for sea_i in sea.subset[1:]:
                    for dm_el_i in range(self.natoms):
                        if np.isclose(dm[sea.subset[0], dm_el_0], dm[sea_i, dm_el_i], atol=asym_tol):
                            matches.append((sea_i, dm_el_i))
                suum = 0.0
                for idxs in matches:
                    suum += dm[idxs]
                avg_dm = suum / len(matches)
                for idxs in matches:
                    new_dm[idxs[0],idxs[1]] = avg_dm
                    new_dm[idxs[1],idxs[0]] = avg_dm
        
        # Transform back to Cartesian coordinates, shout out to @Legendre17 on Math Stack Exchange
        M = np.array([[(new_dm[0,j]**2 + new_dm[i,0]**2 - new_dm[i,j]**2)/2 for j in range(self.natoms)] for i in range(self.natoms)])
        evals, evecs = np.linalg.eigh(M)
        evalMat = np.zeros((self.natoms, self.natoms))
        for i in range(self.natoms):
            evalMat[i,i] = np.sqrt(evals[i])
        X = np.dot(evecs, evalMat)
        self.coords = X[:,-3:]

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
    if A.tol >= B.tol:
        eq_tol = A.tol
    else:
        eq_tol = B.tol
    h = []
    for i in range(A.natoms):
        for j in range(B.natoms):
            if A.masses[i] == B.masses[j]:
                zs = abs(A.coords[i,:]-B.coords[j,:])
                if np.allclose(zs, [0,0,0], atol=eq_tol):
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
