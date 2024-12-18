"""
This code generates the rotation matrices for real spherical harmonics by a recursion relation
found in "Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion."
J. Ivanic and K. Rudenberg: doi/10.1021/jp953350u
"""

import numpy as np
import copy

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
    Rsh.append(np.eye(3))
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
        Rsh[1][[0, 1, 2]] = rot[[2, 0, 1]] 
         
    for r, rsh in enumerate(Rsh):
        if r > 1:
            beeb = generateshuffle(r)
            Rsh[r][[x for x in range(0, 2*r + 1)]] = Rsh[r][[beeb]]
    return Rsh

#generates u, w, and v coefficients, eq. 8.1, found in Table 1 of reference
def UWVCoefficient(l, m1, m2):
    delta = bool(0 == m1)
    if abs(m2) < l:
        denom = (l + m2)*(l - m2)
    else:
        denom = (2.0*l)*(2.0*l - 1.0)
    #print(f"delta denom {delta} {denom} {l} {m1}")
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


def psi4_adapt(rot):
    new = np.zeros((3,3))
    new[:, [0, 1, 2]] = rot[:, [2, 1, 0]]
    new[[0, 1, 2], :] = new[[2, 1, 0], :]
    return new

