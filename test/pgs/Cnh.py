import numpy as np
from numpy.linalg import matrix_power
from molsym.symtext.symtext import Symel
from molsym.symtools import Cn, reflection_matrix, inversion_matrix, Sn

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

C2hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)), Symel("i",None,inversion_matrix()), Symel("C_2^1",z,Cn(z,2))]
C3hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)),
    Symel("C_3^1",z,Cn(z,3)), Symel("C_3^2",z,matrix_power(Cn(z,3),2)), Symel("S_3^1",z,Sn(z,3)), Symel("S_3^5",z,matrix_power(Sn(z,3),5))]
C4hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)), Symel("i",None,inversion_matrix()),
    Symel("C_4^1",z,Cn(z,4)), Symel("C_2^1",z,Cn(z,2)), Symel("C_4^3",z,matrix_power(Cn(z,4),3)), Symel("S_4^1",z,Sn(z,4)), Symel("S_4^3",z,matrix_power(Sn(z,4),3))]
C5hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)),
    Symel("C_5^1",z,Cn(z,5)), Symel("C_5^2",z,matrix_power(Cn(z,5),2)), Symel("C_5^3",z,matrix_power(Cn(z,5),3)), Symel("C_5^4",z,matrix_power(Cn(z,5),4)),
    Symel("S_5^1",z,Sn(z,5)), Symel("S_5^7",z,matrix_power(Sn(z,5),7)), Symel("S_5^3",z,matrix_power(Sn(z,5),3)), Symel("S_5^9",z,matrix_power(Sn(z,5),9))]
C6hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)), Symel("i",None,inversion_matrix()),
    Symel("C_6^1",z,Cn(z,6)), Symel("C_3^1",z,Cn(z,3)), Symel("C_2^1",z,Cn(z,2)), Symel("C_3^2",z,matrix_power(Cn(z,3),2)), Symel("C_6^5",z,matrix_power(Cn(z,6),5)),
    Symel("S_6^1",z,Sn(z,6)), Symel("S_3^1",z,Sn(z,3)), Symel("S_3^5",z,matrix_power(Sn(z,3),5)), Symel("S_6^5",z,matrix_power(Sn(z,6),5))]

C2hirr = ["Ag", "Bg", "Au", "Bu"]
C2hcn = ["E","C_2","i","sigma_h"]
C2hct = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]])

eps = np.exp(2*np.pi*1j/3)
C3hirr = ["A'", "E_1'", "E_2'", "A''", "E_1''", "E_2''"]
C3hcn = ["E","C_3","C_3^2","sigma_h","S_3","S_3^5"]
C3hct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, eps, eps**2, 1.0, eps, eps**2], [1,eps**2,eps, 1,eps**2,eps],
                [1.0, 1.0, 1.0, -1.0, -1.0, -1.0], [1.0, eps, eps**2, -1.0, -eps, -eps**2], [1,eps**2,eps, -1,-eps**2,-eps]])

C4hirr = ["Ag", "Bg", "E_1g", "E_2g", "Au", "Bu", "E_1u", "E_2u"]
C4hcn = ["E","C_4","C_2","C_4^3","i","S_4^3","sigma_h","S_4"]
C4hct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                  [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
                  [1,1j,-1,-1j,1,1j,-1,-1j],
                  [1,-1j,-1,1j,1,-1j,-1,1j],
                  [1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
                  [1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0],
                  [1,1j,-1,-1j,-1,-1j,1,1j],
                  [1,-1j,-1,1j,-1,1j,1,-1j]])

eps = np.exp(2*np.pi*1j/5)
C5hirr = ["A'", "E1_1'", "E1_2'", "E2_1'", "E2_2'", "A''", "E1_1''", "E1_2''", "E2_1''", "E2_2''"]
C5hcn = ["E","C_5","C_5^2","C_5^3","C_5^4","sigma_h","S_5","S_5^7","S_5^3","S_5^9"]
C5hct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [1,eps,eps**2,eps**3,eps**4,1,eps,eps**2,eps**3,eps**4],
         [1,eps**4,eps**3,eps**2,eps,1,eps**4,eps**3,eps**2,eps],
         [1,eps**2,eps**4,eps,eps**3,1,eps**2,eps**4,eps,eps**3],
         [1,eps**3,eps,eps**4,eps**2,1,eps**3,eps,eps**4,eps**2],
         [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
         [1,eps,eps**2,eps**3,eps**4,-1,-eps,-eps**2,-eps**3,-eps**4],
         [1,eps**4,eps**3,eps**2,eps,-1,-eps**4,-eps**3,-eps**2,-eps],
         [1,eps**2,eps**4,eps,eps**3,-1,-eps**2,-eps**4,-eps,-eps**3],
         [1,eps**3,eps,eps**4,eps**2,-1,-eps**3,-eps,-eps**4,-eps**2]])
         
eps = np.exp(2*np.pi*1j/6)
C6hirr = ["Ag", "Bg", "E1_1g", "E1_2g", "E2_1g", "E2_2g", "Au", "Bu", "E1_1u", "E1_2u", "E2_1u", "E2_2u"]
C6hcn = ["E","C_6","C_3","C_2","C_3^2","C_6^5","i","S_3^5","S_6^5","sigma_h","S_6","S_3"]
C6hct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
         [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
         [1,eps,eps**2,-1,eps**4,eps**5,1,eps,eps**2,-1,eps**4,eps**5],
         [1,eps**5,eps**4,-1,eps**2,eps,1,eps**5,eps**4,-1,eps**2,eps],
         [1,eps**2,eps**4,1,eps**2,eps**4,1,eps**2,eps**4,1,eps**2,eps**4],
         [1,eps**4,eps**2,1,eps**4,eps**2,1,eps**4,eps**2,1,eps**4,eps**2],
         [1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
         [1.0, -1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
         [1,eps,eps**2,-1,eps**4,eps**5,-1,-eps,-eps**2,1,-eps**4,-eps**5],
         [1,eps**5,eps**4,-1,eps**2,eps,-1,-eps**5,-eps**4,1,-eps**2,-eps],
         [1,eps**2,eps**4,1,eps**2,eps**4,-1,-eps**2,-eps**4,-1,-eps**2,-eps**4],
         [1,eps**4,eps**2,1,eps**4,eps**2,-1,-eps**4,-eps**2,-1,-eps**4,-eps**2]])
