import numpy as np
#import psi4
from math import isclose
from molsym.molecule import *
from molsym.symtools import *

def find_point_group(mol):
    paxis = [0,0,0]
    saxis = [0,0,0]
    moit = calcmoit(mol)
    Ia_mol, Ib_mol, Ic_mol = np.sort(np.linalg.eigh(moit)[0])
    if Ia_mol == 0.0:
        if isequivalent(mol, mol.transform(inversion_matrix())):
            pg = "D0h"
        else:
            pg = "C0v"
    elif isclose(Ia_mol,Ib_mol,abs_tol=mol.tol) and isclose(Ia_mol, Ic_mol, abs_tol=mol.tol):
        seas = mol.find_SEAs()
        n, axes = num_C2(mol, seas)
        invertable = isequivalent(mol, mol.transform(inversion_matrix()))
        if n == 15:
            # paxis is any C2 axis, saxis is defined as the C2 axis that is coplanar with paxis and the nearest C3 axis to paxis
            paxis = axes[0]
            c3s = find_C3s_for_Ih(mol)
            for c3 in c3s:
                if isclose(np.arccos(abs(np.dot(c3, paxis))), 0.36486382647383764, abs_tol = 1e-4):
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
            if invertable:
                pg = "Ih"
            else:
                pg = "I"
        
        elif n == 9:
            # paxis and saxis are orthogonal C4 axes
            c4s = find_C4s_for_Oh(mol)
            paxis = c4s[0]
            saxis = c4s[1]
            if invertable:
                pg = "Oh"
            else:
                pg = "O"
        elif n == 3:
            paxis = axes[0]
            saxis = axes[1]
            if invertable:
                pg = "Th"
            else:
                # No path for T!!! TODO
                pg = "Td"
    else:
        seas = mol.find_SEAs()
        rot_set = find_rotation_sets(mol, seas)
        rots = find_rotations(mol, rot_set)
        if len(rots) < 1:
            c2 = find_a_c2(mol, seas)
            if c2 is not None:
                paxis = c2
                c2_ortho_chk, c2_ortho = is_there_ortho_c2(mol, seas, paxis)
                sigmav_chk, sigmav = is_there_sigmav(mol, seas, paxis)
                sigmah_chk = is_there_sigmah(mol, c2)
                if c2_ortho_chk:
                    saxis = c2_ortho
                    if sigmah_chk:
                        pg = "D2h"
                    else:
                        if sigmav_chk:
                            pg = "D2d"
                        else:
                            pg = "D2"
                else:
                    if sigmah_chk:
                        pg = "C2h"
                    else:
                        if sigmav_chk:
                            if sigmav.any():
                                saxis = np.cross(paxis,sigmav)
                            pg = "C2v"
                        else:
                            S4 = Sn(c2, 4)
                            molB = mol.transform(S4)
                            chk = isequivalent(mol, molB)
                            if chk:
                                pg = "S4"
                            else:
                                pg = "C2"
            else:
                molB = mol.transform(inversion_matrix())
                if isequivalent(mol, molB):
                    pg = "Ci"
                else:
                    sigmav_chk, sigmav = is_there_sigmav(mol, seas, np.asarray([0,0,0]))
                    if sigmav_chk:
                        if sigmav is not None:
                            paxis = sigmav
                        pg = "Cs"
                    else:
                        pg = "C1"
        else:
            rots = find_rotations(mol, rot_set)
            Cn = highest_order_axis(rots)
            paxis = rots[0].axis
            ortho_c2_chk, c2_ortho = is_there_ortho_c2(mol, seas, paxis)
            sigmah_chk = is_there_sigmah(mol, paxis)
            sigmav_chk, sigmav = is_there_sigmav(mol, seas, paxis)
            if ortho_c2_chk:
                saxis = c2_ortho
                if sigmah_chk:
                    pg = "D"+str(Cn)+"h"
                elif sigmav_chk:
                    pg = "D"+str(Cn)+"d"
                else:
                    pg = "D"+str(Cn)
            elif sigmah_chk:
                pg = "C"+str(Cn)+"h"
            elif sigmav_chk:
                if sigmav.any():
                    saxis = normalize(np.cross(paxis,sigmav))
                pg = "C"+str(Cn)+"v"
            else:
                S2n = Sn(paxis, Cn*2)
                molB = mol.transform(S2n)
                if isequivalent(mol, molB):
                    pg = "S"+str(2*Cn)
                else:
                    pg = "C"+str(Cn)
    return pg, (paxis, saxis)
