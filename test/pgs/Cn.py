import numpy as np
from numpy.linalg import matrix_power
from molsym.symtools import Cn, reflection_matrix, inversion_matrix
from molsym.symtext.symel import Symel

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

C1s = [Symel("E",None,np.identity(3))]
Cis = [Symel("E",None,np.identity(3)), Symel("i",None,inversion_matrix())]
Css = [Symel("E",z,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z))]

C2s = [Symel("E",z,np.identity(3)), Symel("C_2^1",z,Cn(z,2))]
C3s = [Symel("E",z,np.identity(3)), Symel("C_3^1",z,Cn(z,3)),Symel("C_3^2",z,matrix_power(Cn(z,3),2))]
C4s = [Symel("E",z,np.identity(3)), Symel("C_4^1",z,Cn(z,4)),Symel("C_2^1",z,Cn(z,2)),Symel("C_4^3",z,matrix_power(Cn(z,4),3))]
C5s = [Symel("E",z,np.identity(3)), Symel("C_5^1",z,Cn(z,5)),Symel("C_5^2",z,matrix_power(Cn(z,5),2)),Symel("C_5^3",z,matrix_power(Cn(z,5),3)),Symel("C_5^4",z,matrix_power(Cn(z,5),4))]
C6s = [Symel("E",z,np.identity(3)), Symel("C_6^1",z,Cn(z,6)),Symel("C_3^1",z,Cn(z,3)),Symel("C_2^1",z,Cn(z,2)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)),Symel("C_6^5",z,matrix_power(Cn(z,6),5))]

C1irr = ["A"]
C1cn = ["E"]
C1ct = np.array([1.0])

C2irr = ["A","B"]
C2cn = ["E","C_2"]
C2ct = np.array([[1.0,1.0],[1.0,-1.0]])

eps = np.exp(2*np.pi*1j/3)
C3irr = ["A","E_1","E_2"]
C3cn = ["E","C_3","C_3^2"]
C3ct = np.array([[1.0,1.0,1.0],[1,eps,eps**2],[1,eps**2,eps]])

C4irr = ["A","B","E_1","E_2"]
C4cn = ["E","C_4","C_2","C_4^3"]
C4ct = np.array([[1.0,1.0,1.0,1.0],[1.0,-1.0,1.0,-1.0],[1,1j,-1,-1j],[1,-1j,-1,1j]])

eps = np.exp(2*np.pi*1j/5)
C5irr = ["A","E1_1","E1_2","E2_1","E2_2"]
C5cn = ["E","C_5","C_5^2","C_5^3","C_5^4"]
C5ct = np.array([[1.0,1.0,1.0,1.0,1.0],[1,eps,eps**2,eps**3,eps**4],[1,eps**4,eps**3,eps**2,eps],[1,eps**2,eps**4,eps,eps**3],[1,eps**3,eps,eps**4,eps**2]])

eps = np.exp(2*np.pi*1j/6)
C6irr = ["A","B","E1_1","E1_2","E2_1","E2_2"]
C6cn = ["E","C_6","C_3","C_2","C_3^2","C_6^5"]
C6ct = np.array([[1.0,1.0,1.0,1.0,1.0,1.0],[1.0,-1.0,1.0,-1.0,1.0,-1.0],
                 [1,eps,eps**2,-1,eps**4,eps**5],
                 [1,eps**5,eps**4,-1,eps**2,eps],
                 [1,eps**2,eps**4,1,eps**2,eps**4],
                 [1,eps**4,eps**2,1,eps**4,eps**2]])
