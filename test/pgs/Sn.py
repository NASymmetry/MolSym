import numpy as np
from numpy.linalg import matrix_power
from molsym.molecule import Cn, reflection_matrix, inversion_matrix, Sn
from molsym.symtext.symtext import Symel

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

rt2 = np.sqrt(2)
c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

S4s = [Symel("E",None,np.identity(3)), Symel("C_2^1",z,Cn(z,2)),
        Symel("S_4^1",z,Sn(z,4)), Symel("S_4^3",z,matrix_power(Sn(z,4),3))]
S6s = [Symel("E",None,np.identity(3)), Symel("i",None,inversion_matrix()), 
        Symel("C_3^1",z,Cn(z,3)), Symel("C_3^2",z,matrix_power(Cn(z,3),2)),
        Symel("S_6^1",z,Sn(z,6)), Symel("S_6^5",z,matrix_power(Sn(z,6),5))]
S8s = [Symel("E",None,np.identity(3)),
        Symel("C_4^1",z,Cn(z,4)),Symel("C_2^1",z,Cn(z,2)),Symel("C_4^3",z,matrix_power(Cn(z,4),3)),
        Symel("S_8^1",z,Sn(z,8)), Symel("S_8^3",z,matrix_power(Sn(z,8),3)), Symel("S_8^5",z,matrix_power(Sn(z,8),5)), Symel("S_8^7",z,matrix_power(Sn(z,8),7))]
S10s = [Symel("E",None,np.identity(3)), Symel("i",None,inversion_matrix()),
        Symel("C_5^1",z,Cn(z,5)),Symel("C_5^2",z,matrix_power(Cn(z,5),2)),Symel("C_5^3",z,matrix_power(Cn(z,5),3)), Symel("C_5^4",z,matrix_power(Cn(z,5),4)),
        Symel("S_10^1",z,Sn(z,10)), Symel("S_10^3",z,matrix_power(Sn(z,10),3)), Symel("S_10^7",z,matrix_power(Sn(z,10),7)), Symel("S_10^9",z,matrix_power(Sn(z,10),9))]

S4irr = ["A", "B", "E"]
S4cn = ["E","S_4","C_2","S_4^3"]
S4ct = np.array([[1.0,1.0,1.0,1.0],[1.0,-1.0,1.0,-1.0],[2.0,0.0,-2.0,0.0]])
S6irr = ["Ag", "Eg", "Au", "Eu"]
S6cn = ["E","C_3","C_3^2","i","S_6^5","S_6"]
S6ct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2.0, -1.0, -1.0, 2.0, -1.0, -1.0], 
                [1.0, 1.0, 1.0, -1.0, -1.0, -1.0], [2.0, -1.0, -1.0, -2.0, 1.0, 1.0]])
S8irr = ["A", "B", "E1", "E2", "E3"]
S8cn = ["E","S_8","C_4","S_8^3","C_2","S_8^5","C_4^3","S_8^7"]
S8ct =np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
                [2.0,  rt2,  0.0, -rt2, -2.0, -rt2,  0.0,  rt2],
                [2.0,  0.0, -2.0,  0.0,  2.0,  0.0, -2.0,  0.0],
                [2.0, -rt2,  0.0,  rt2, -2.0,  rt2,  0.0, -rt2]])
S10irr = ["Ag", "E1g", "E2g", "Au", "E1u", "E2u"]
S10cn = ["E","C_5","C_5^2","C_5^3","C_5^4","i","S_10^7","S_10^9","S_10","S_10^3"]
S10ct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [2.0, c52, c54, c54, c52, 2.0, c52, c54, c54, c52],
         [2.0, c54, c52, c52, c54, 2.0, c54, c52, c52, c54],
         [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
         [2.0, c52, c54, c54, c52, -2.0, -c52, -c54, -c54, -c52],
         [2.0, c54, c52, c52, c54, -2.0, -c54, -c52, -c52,  -c54]])
