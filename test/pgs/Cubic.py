import numpy as np
from numpy.linalg import matrix_power
from molsym.symtools import Cn, Sn, reflection_matrix, inversion_matrix, normalize
from molsym.symtext.symel import Symel

x,y,z = np.eye(3,dtype=float)
a_vec = normalize(np.array([ 1.0,  1.0,  1.0]))
b_vec = normalize(np.array([-1.0,  1.0, -1.0]))
g_vec = normalize(np.array([-1.0, -1.0,  1.0]))
d_vec = normalize(np.array([ 1.0, -1.0, -1.0]))

Ts = [Symel("E", None, np.eye(3)),
      Symel("C_3(alpha)", a_vec, Cn(a_vec, 3)),
      Symel("C_3^2(alpha)", a_vec, matrix_power(Cn(a_vec, 3), 2)),
      Symel("C_3(beta)", b_vec, Cn(b_vec, 3)),
      Symel("C_3^2(beta)", b_vec, matrix_power(Cn(b_vec, 3), 2)),
      Symel("C_3(gamma)", g_vec, Cn(g_vec, 3)),
      Symel("C_3^2(gamma)", g_vec, matrix_power(Cn(g_vec, 3), 2)),
      Symel("C_3(delta)", d_vec, Cn(d_vec, 3)),
      Symel("C_3^2(delta)", d_vec, matrix_power(Cn(d_vec, 3), 2)),
      Symel("C_2(x)", x, Cn(x, 2)),
      Symel("C_2(y)", y, Cn(y, 2)),
      Symel("C_2(z)", z, Cn(z, 2))]

xyp = normalize(np.array([1.0, 1.0, 0.0]))
xym = normalize(np.array([1.0,-1.0, 0.0]))
xzp = normalize(np.array([1.0, 0.0, 1.0]))
xzm = normalize(np.array([1.0, 0.0,-1.0]))
yzp = normalize(np.array([0.0, 1.0, 1.0]))
yzm = normalize(np.array([0.0, 1.0,-1.0]))
namelist = ["xyp","xym","xzp","xzm","yzp","yzm"]

Tds = Ts + [
    Symel("sigma_d(xyp)", xyp, reflection_matrix(xyp)),
    Symel("sigma_d(xym)", xym, reflection_matrix(xym)),
    Symel("sigma_d(xzp)", xzp, reflection_matrix(xzp)),
    Symel("sigma_d(xzm)", xzm, reflection_matrix(xzm)),
    Symel("sigma_d(yzp)", yzp, reflection_matrix(yzp)),
    Symel("sigma_d(yzm)", yzm, reflection_matrix(yzm)),
    Symel("S_4(x)", x, Sn(x, 4)),
    Symel("S_4^3(x)", x, matrix_power(Sn(x, 4), 3)),
    Symel("S_4(y)", y, Sn(y, 4)),
    Symel("S_4^3(y)", y, matrix_power(Sn(y, 4), 3)),
    Symel("S_4(z)", z, Sn(z, 4)),
    Symel("S_4^3(z)", z, matrix_power(Sn(z, 4), 3)),
]

Ths = Ts + [
      Symel("i", None, -np.eye(3)),
      Symel("S_6(alpha)", a_vec, Sn(a_vec, 6)),
      Symel("S_6^5(alpha)", a_vec, matrix_power(Sn(a_vec, 6), 5)),
      Symel("S_6(beta)", b_vec, Sn(b_vec, 6)),
      Symel("S_6^5(beta)", b_vec, matrix_power(Sn(b_vec, 6), 5)),
      Symel("S_6(gamma)", g_vec, Sn(g_vec, 6)),
      Symel("S_6^5(gamma)", g_vec, matrix_power(Sn(g_vec, 6), 5)),
      Symel("S_6(delta)", d_vec, Sn(d_vec, 6)),
      Symel("S_6^5(delta)", d_vec, matrix_power(Sn(d_vec, 6), 5)),
      Symel("sigma_h(x)", x, reflection_matrix(x)),
      Symel("sigma_h(y)", y, reflection_matrix(y)),
      Symel("sigma_h(z)", z, reflection_matrix(z))
]

a_vec = normalize(np.array([ 1.0,  1.0,  1.0]))
b_vec = normalize(np.array([ 1.0, -1.0,  1.0]))
g_vec = normalize(np.array([ 1.0,  1.0, -1.0]))
d_vec = normalize(np.array([ 1.0, -1.0, -1.0]))

Os = [
    Symel("E", None, np.eye(3)),
    Symel("C_4(x)", x, Cn(x, 4)),
    Symel("C_2(x)", x, matrix_power(Cn(x, 4),2)),
    Symel("C_4^3(x)", x, matrix_power(Cn(x, 4),3)),
    Symel("C_4(y)", y, Cn(y, 4)),
    Symel("C_2(y)", y, matrix_power(Cn(y, 4),2)),
    Symel("C_4^3(y)", y, matrix_power(Cn(y, 4),3)),
    Symel("C_4(z)", z, Cn(z, 4)),
    Symel("C_2(z)", z, matrix_power(Cn(z, 4),2)),
    Symel("C_4^3(z)", z, matrix_power(Cn(z, 4),3)),
    Symel("C_3(alpha)", a_vec, Cn(a_vec, 3)),
    Symel("C_3^2(alpha)", a_vec, matrix_power(Cn(a_vec, 3),2)),
    Symel("C_3(beta)", b_vec, Cn(b_vec, 3)),
    Symel("C_3^2(beta)", b_vec, matrix_power(Cn(b_vec, 3),2)),
    Symel("C_3(gamma)", g_vec, Cn(g_vec, 3)),
    Symel("C_3^2(gamma)", g_vec, matrix_power(Cn(g_vec, 3),2)),
    Symel("C_3(delta)", d_vec, Cn(d_vec, 3)),
    Symel("C_3^2(delta)", d_vec, matrix_power(Cn(d_vec, 3),2)),
    Symel("C_2(xyp)", xyp, Cn(xyp, 2)),
    Symel("C_2(xym)", xym, Cn(xym, 2)),
    Symel("C_2(xzp)", xzp, Cn(xzp, 2)),
    Symel("C_2(xzm)", xzm, Cn(xzm, 2)),
    Symel("C_2(yzp)", yzp, Cn(yzp, 2)),
    Symel("C_2(yzm)", yzm, Cn(yzm, 2))
]

Ohs = Os + [
    Symel("i", None, -np.eye(3)),
    Symel("S_4(x)", x, Sn(x, 4)),
    Symel("S_4^3(x)", x, matrix_power(Sn(x, 4), 3)),
    Symel("S_4(y)", y, Sn(y, 4)),
    Symel("S_4^3(y)", y, matrix_power(Sn(y, 4), 3)),
    Symel("S_4(z)", z, Sn(z, 4)),
    Symel("S_4^3(z)", z, matrix_power(Sn(z, 4), 3)),
    Symel("sigma_h(x)", x, reflection_matrix(x)),
    Symel("sigma_h(y)", y, reflection_matrix(y)),
    Symel("sigma_h(z)", z, reflection_matrix(z)),
    Symel("S_6(alpha)", a_vec, Sn(a_vec, 6)),
    Symel("S_6^5(alpha)", a_vec, matrix_power(Sn(a_vec, 6),5)),
    Symel("S_6(beta)", b_vec, Sn(b_vec, 6)),
    Symel("S_6^5(beta)", b_vec, matrix_power(Sn(b_vec, 6),5)),
    Symel("S_6(gamma)", g_vec, Sn(g_vec, 6)),
    Symel("S_6^5(gamma)", g_vec, matrix_power(Sn(g_vec, 6),5)),
    Symel("S_6(delta)", d_vec, Sn(d_vec, 6)),
    Symel("S_6^5(delta)", d_vec, matrix_power(Sn(d_vec, 6),5)),
    Symel("sigma_d(xyp)", xyp, reflection_matrix(xyp)),
    Symel("sigma_d(xym)", xym, reflection_matrix(xym)),
    Symel("sigma_d(xzp)", xzp, reflection_matrix(xzp)),
    Symel("sigma_d(xzm)", xzm, reflection_matrix(xzm)),
    Symel("sigma_d(yzp)", yzp, reflection_matrix(yzp)),
    Symel("sigma_d(yzm)", yzm, reflection_matrix(yzm))
]