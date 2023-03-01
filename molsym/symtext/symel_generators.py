from molsym.molecule import *
from .symtext import Symel, reduce
import numpy as np
from numpy.linalg import matrix_power

def generate_Cn(n):
    symels = []
    axis = np.asarray([0,0,1])
    #axis = [0 0 1]'
    cn_r = Cn(axis, n)
    for i in range(1,n):
        a, b = reduce(n, i)
        symels.append(Symel(f"C_{a:d}^{b:d}", matrix_power(cn_r,i))) # Cns
    return symels

#def generate_Sn(n):
#    generate_Sn(n, False)

def generate_Sn(n, S2n=False):
    symels = []
    axis = np.asarray([0,0,1])
    sigma_h = reflection_matrix(axis)
    cn_r = Cn(axis, n)
    if S2n: # Generating improper rotations for S2n PG
        for i in range(1,n):
            if i % 2 == 0:
                continue
            else:
                a, b = reduce(n, i)
                if a == 2:
                    continue
                else:
                    symels.append(Symel(f"S_{a}^{b}", np.dot(matrix_power(cn_r,i),sigma_h)))
        return symels
    for i in range(1,n):
        a, b = reduce(n, i)
        if b % 2 == 0:
            b += a
        if a == 2:
            continue
        else:
            symels.append(Symel(f"S_{a}^{b}", np.dot(matrix_power(cn_r,i),sigma_h))) # Sns
    return symels

def generate_sigma_v(n):
    if n % 2 == 0:
        nsigma_vs = n >> 1
    else:
        nsigma_vs = n
    symels = []
    x_axis = np.asarray([1,0,0]) # Orient C2 and sigma_v along x-axis
    z_axis = np.asarray([0,0,1])
    rot_mat = Cn(z_axis, n)
    for i in range(nsigma_vs):
        axis = np.cross(np.dot(matrix_power(rot_mat,i), x_axis),z_axis)
        symels.append(Symel(f"sigma_v({i+1})", reflection_matrix(axis)))
    return symels

def generate_sigma_d(n):
    symels = []
    x_axis = np.asarray([1,0,0]) # Orient C2 and sigma_v along x-axis
    z_axis = np.asarray([0,0,1])
    rot_mat = Cn(z_axis, 2*n)
    base_axis = np.dot(Cn(z_axis, 4*n),x_axis) # Rotate x-axis by Cn/2 to produce an axis for sigma_d's
    for i in range(n):
        axis = np.cross(np.dot(matrix_power(rot_mat,i), base_axis),z_axis)
        symels.append(Symel(f"sigma_d({i+1})", reflection_matrix(axis)))
    return symels

def generate_C2p(n):
    if n % 2 == 0:
        nn = n >> 1
    else:
        nn = n
    symels = []
    x_axis = np.asarray([1,0,0]) # Orient C2 and sigma_v along x-axis
    rot_mat = Cn([0,0,1], n)
    for i in range(nn):
        axis = np.dot(matrix_power(rot_mat,i), x_axis)
        symels.append(Symel(f"C_2'({i+1})", Cn(axis, 2)))
    return symels

def generate_C2pp(n):
    nn = n >> 1
    symels = []
    x_axis = np.asarray([1,0,0])
    rot_mat = Cn([0,0,1], n)
    base_axis = np.dot(Cn([0,0,1], 2*n),x_axis)
    for i in range(nn):
        axis = np.dot(matrix_power(rot_mat,i), base_axis)
        symels.append(Symel(f"C_2''({i+1})", Cn(axis, 2)))
    return symels

def generate_T():
    pass

def generate_Td():
    pass

def generate_Th():
    pass

def generate_O():
    pass

def generate_Oh():
    pass

def generate_I():
    pass

def generate_Ih():
    pass
