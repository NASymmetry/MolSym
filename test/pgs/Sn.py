import numpy as np
from numpy.linalg import matrix_power
from molsym.symtools import Cn, inversion_matrix, Sn
from molsym.symtext.symel import Symel

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

rt2 = np.sqrt(2)
c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

S4s = [Symel("E",None,np.identity(3)), Symel("C_2",z,Cn(z,2)),
        Symel("S_4",z,Sn(z,4)), Symel("S_4^3",z,matrix_power(Sn(z,4),3))]
S6s = [Symel("E",None,np.identity(3)), Symel("i",None,inversion_matrix()), 
        Symel("C_3",z,Cn(z,3)), Symel("C_3^2",z,matrix_power(Cn(z,3),2)),
        Symel("S_6",z,Sn(z,6)), Symel("S_6^5",z,matrix_power(Sn(z,6),5))]
S8s = [Symel("E",None,np.identity(3)),
        Symel("C_4",z,Cn(z,4)),Symel("C_2",z,Cn(z,2)),Symel("C_4^3",z,matrix_power(Cn(z,4),3)),
        Symel("S_8",z,Sn(z,8)), Symel("S_8^3",z,matrix_power(Sn(z,8),3)), Symel("S_8^5",z,matrix_power(Sn(z,8),5)), Symel("S_8^7",z,matrix_power(Sn(z,8),7))]
S10s = [Symel("E",None,np.identity(3)), Symel("i",None,inversion_matrix()),
        Symel("C_5",z,Cn(z,5)),Symel("C_5^2",z,matrix_power(Cn(z,5),2)),Symel("C_5^3",z,matrix_power(Cn(z,5),3)), Symel("C_5^4",z,matrix_power(Cn(z,5),4)),
        Symel("S_10",z,Sn(z,10)), Symel("S_10^3",z,matrix_power(Sn(z,10),3)), Symel("S_10^7",z,matrix_power(Sn(z,10),7)), Symel("S_10^9",z,matrix_power(Sn(z,10),9))]

S4irr = ["A", "B", "E_1", "E_2"]
S4cn = ["E","S_4","C_2","S_4^3"]
S4ct = np.array([[1.0,1.0,1.0,1.0],[1.0,-1.0,1.0,-1.0],[1,1j,-1,-1j],[1,-1j,-1,1j]])

eps = np.exp(2*np.pi*1j/3)
S6irr = ["Ag", "E_1g", "E_2g", "Au", "E_1u", "E_2u"]
S6cn = ["E","C_3","C_3^2","i","S_6^5","S_6"]
S6ct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                 [1.0, eps, eps**2, 1.0, eps, eps**2], 
                 [1,eps**2,eps, 1,eps**2,eps],
                 [1.0, 1.0, 1.0, -1.0, -1.0, -1.0], 
                 [1.0, eps, eps**2, -1.0, -eps, -eps**2], 
                 [1,eps**2,eps, -1,-eps**2,-eps]])

eps = np.exp(2*np.pi*1j/8)
S8irr = ["A", "B", "E1_1", "E1_2", "E2_1", "E2_2", "E3_1", "E3_2"]
S8cn = ["E","S_8","C_4","S_8^3","C_2","S_8^5","C_4^3","S_8^7"]
S8ct =np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
                [1, eps, 1j, eps**3, -1, eps**5, -1j, eps**7],
                [1, eps**7, -1j, eps**5, -1, eps**3, 1j, eps],
                [1, 1j, -1, -1j, 1, 1j, -1, -1j],
                [1, -1j, -1, 1j, 1, -1j, -1, 1j],
                [1, eps**3, -1j, eps, -1, eps**7, 1j, eps**5],
                [1, eps**5, 1j, eps**7, -1, eps, -1j, eps**3]])

eps = np.exp(2*np.pi*1j/5)
S10irr = ["Ag", "E1_1g", "E1_2g", "E2_1g", "E2_2g", "Au", "E1_1u", "E1_2u", "E2_1u", "E2_2u"]
S10cn = ["E","C_5","C_5^2","C_5^3","C_5^4","i","S_10^7","S_10^9","S_10","S_10^3"]
S10ct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [1,eps,eps**2,eps**3,eps**4,1,eps,eps**2,eps**3,eps**4],
         [1,eps**4,eps**3,eps**2,eps,1,eps**4,eps**3,eps**2,eps],
         [1,eps**2,eps**4,eps,eps**3,1,eps**2,eps**4,eps,eps**3],
         [1,eps**3,eps,eps**4,eps**2,1,eps**3,eps,eps**4,eps**2],
         [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
         [1,eps,eps**2,eps**3,eps**4,-1,-eps,-eps**2,-eps**3,-eps**4],
         [1,eps**4,eps**3,eps**2,eps,-1,-eps**4,-eps**3,-eps**2,-eps],
         [1,eps**2,eps**4,eps,eps**3,-1,-eps**2,-eps**4,-eps,-eps**3],
         [1,eps**3,eps,eps**4,eps**2,-1,-eps**3,-eps,-eps**4,-eps**2]])