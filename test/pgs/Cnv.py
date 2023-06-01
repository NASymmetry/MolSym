import numpy as np
from numpy.linalg import matrix_power
from molsym.molecule import Cn, reflection_matrix, inversion_matrix, Sn
from molsym.symtext.symtext import Symel

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

C2vs = [Symel("E",None,np.identity(3)), Symel("C_2^1",z,Cn(z,2)), Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_d(1)",x,reflection_matrix(x))]
C3vs = [Symel("E",None,np.identity(3)), Symel("C_3^1",z,Cn(z,3)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",np.dot(Cn(z,3),y.T),reflection_matrix(np.dot(Cn(z,3),y.T))), Symel("sigma_v(3)",np.dot(matrix_power(Cn(z,3),2),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,3),2),y.T)))]
C4vs = [Symel("E",None,np.identity(3)), Symel("C_4^1",z,Cn(z,4)),Symel("C_2^1",z,Cn(z,2)),Symel("C_4^3",z,matrix_power(Cn(z,4),3)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",x,reflection_matrix(x)), Symel("sigma_d(1)",np.dot(Cn(z,8),y.T),reflection_matrix(np.dot(Cn(z,8),y.T))), Symel("sigma_d(2)",np.dot(Cn(z,8),x.T),reflection_matrix(np.dot(Cn(z,8),x.T)))]
C5vs = [Symel("E",None,np.identity(3)), Symel("C_5^1",z,Cn(z,5)),Symel("C_5^2",z,matrix_power(Cn(z,5),2)),Symel("C_5^3",z,matrix_power(Cn(z,5),3)),Symel("C_5^4",z,matrix_power(Cn(z,5),4)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",np.dot(Cn(z,5),y.T),reflection_matrix(np.dot(Cn(z,5),y.T))), Symel("sigma_v(3)",np.dot(matrix_power(Cn(z,5),2),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,5),2),y.T))), 
        Symel("sigma_v(4)",np.dot(matrix_power(Cn(z,5),3),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,5),3),y.T))), Symel("sigma_v(5)",np.dot(matrix_power(Cn(z,5),4),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,5),4),y.T)))]
C6vs = [Symel("E",None,np.identity(3)), Symel("C_6^1",z,Cn(z,6)),Symel("C_3^1",z,Cn(z,3)),Symel("C_2^1",z,Cn(z,2)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)),Symel("C_6^5",z,matrix_power(Cn(z,6),5)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",np.dot(Cn(z,6),y.T),reflection_matrix(np.dot(Cn(z,6),y.T))), Symel("sigma_v(3)",np.dot(matrix_power(Cn(z,6),2),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,6),2),y.T))),
        Symel("sigma_d(1)",np.dot(matrix_power(Cn(z,6),2),x.T),reflection_matrix(np.dot(matrix_power(Cn(z,6),2),x.T))), Symel("sigma_d(2)",x,reflection_matrix(x)), Symel("sigma_d(3)",np.dot(Cn(z,6),x.T),reflection_matrix(np.dot(Cn(z,6),x.T)))]

C2virr = ["A1", "A2", "B1", "B2"]
C2vcn = ["E","C_2","sigma_v(xz)","sigma_d(yz)"]
C2vct = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]])

C3virr = ["A1", "A2", "E"]
C3vcn = ["E","2C_3","3sigma_v"]
C3vct = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [2.0, -1.0, 0.0]])

C4virr = ["A1", "A2", "B1", "B2", "E"]
C4vcn = ["E","2C_4","C_2","2sigma_v","2sigma_d"]
C4vct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0], [2.0, 0.0, -2.0, 0.0, 0.0]])

C5virr = ["A1", "A2", "E1", "E2"]
C5vcn = ["E","2C_5","2C_5^2","5sigma_v"]
C5vct = np.array([[1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, -1.0],
         [2.0, c52, c54, 0.0],
         [2.0, c54, c52, 0.0]])

C6virr = ["A1", "A2", "B1", "B2", "E1", "E2"]
C6vcn = ["E","2C_6","2C_3","C_2","3sigma_v","3sigma_d"]
C6vct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0, -1.0,  1.0],
        [2.0,  1.0, -1.0, -2.0,  0.0,  0.0],
        [2.0, -1.0, -1.0,  2.0,  0.0,  0.0]])
