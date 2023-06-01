import numpy as np
from numpy.linalg import matrix_power
from molsym.molecule import Cn, reflection_matrix, inversion_matrix, Sn
from molsym.symtext.symtext import Symel

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

D2s = [Symel("E",None,np.identity(3)), Symel("C_2^1",z,Cn(z,2)), Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2''(1)",y,Cn(y,2))]
D3s = [Symel("E",None,np.identity(3)), Symel("C_3^1",z,Cn(z,3)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)),
       Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,3),x.T),Cn(np.dot(Cn(z,3),x.T),2)), Symel("C_2'(3)",np.dot(matrix_power(Cn(z,3),2),x.T),Cn(np.dot(matrix_power(Cn(z,3),2),x.T),2))]
D4s = [Symel("E",None,np.identity(3)), Symel("C_4^1",z,Cn(z,4)),Symel("C_2^1",z,Cn(z,2)),Symel("C_4^3",z,matrix_power(Cn(z,4),3)),
       Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,4),x.T),Cn(np.dot(Cn(z,4),x.T),2)), Symel("C_2''(1)",np.dot(matrix_power(Cn(z,8),1),x.T),Cn(np.dot(matrix_power(Cn(z,8),1),x.T),2)), Symel("C_2''(2)",np.dot(matrix_power(Cn(z,8),3),x.T),Cn(np.dot(matrix_power(Cn(z,8),3),x.T),2))]
D5s = [Symel("E",None,np.identity(3)), Symel("C_5^1",z,Cn(z,5)),Symel("C_5^2",z,matrix_power(Cn(z,5),2)),Symel("C_5^3",z,matrix_power(Cn(z,5),3)), Symel("C_5^4",z,matrix_power(Cn(z,5),4)),
       Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,5),x.T),Cn(np.dot(Cn(z,5),x.T),2)), Symel("C_2'(3)",np.dot(matrix_power(Cn(z,5),2),x.T),Cn(np.dot(matrix_power(Cn(z,5),2),x.T),2)), Symel("C_2'(4)",np.dot(matrix_power(Cn(z,5),3),x.T),Cn(np.dot(matrix_power(Cn(z,5),3),x.T),2)), Symel("C_2'(5)",np.dot(matrix_power(Cn(z,5),4),x.T),Cn(np.dot(matrix_power(Cn(z,5),4),x.T),2))]
D6s = [Symel("E",None,np.identity(3)), Symel("C_6^1",z,Cn(z,6)),Symel("C_3^1",z,Cn(z,3)),Symel("C_2^1",z,Cn(z,2)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)), Symel("C_6^5",z,matrix_power(Cn(z,6),5)),
       Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,6),x.T),Cn(np.dot(Cn(z,6),x.T),2)), Symel("C_2'(3)",np.dot(Cn(z,3),x.T),Cn(np.dot(Cn(z,3),x.T),2)),
       Symel("C_2''(1)",np.dot(Cn(z,12),x.T),Cn(np.dot(Cn(z,12),x.T),2)), Symel("C_2''(2)",np.dot(matrix_power(Cn(z,12),3),x.T),Cn(np.dot(matrix_power(Cn(z,12),3),x.T),2)), Symel("C_2''(3)",np.dot(matrix_power(Cn(z,12),5),x.T),Cn(np.dot(matrix_power(Cn(z,12),5),x.T),2))]

D2irr = ["A", "B1", "B2", "B3"]
D2cn = ["E","C_2(z)","C_2(y)","C_2(x)"]
D2ct = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]])

D3irr = ["A1", "A2", "E"]
D3cn = ["E","2C_3","3C_2"]
D3ct = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [2.0, -1.0, 0.0]])

D4irr = ["A1", "A2", "B1", "B2", "E"]
D4cn = ["E","2C_4","C_2","2C_2'","2C_2''"]
D4ct = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0], [2.0, 0.0, -2.0, 0.0, 0.0]])

D5irr = ["A1", "A2", "E1", "E2"]
D5cn = ["E","2C_5","2C_5^2","5C_2"]
D5ct = np.array([[1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, -1.0],
         [2.0, c52, c54, 0.0],
         [2.0, c54, c52, 0.0]])

D6irr = ["A1", "A2", "B1", "B2", "E1", "E2"]
D6cn = ["E","2C_6","2C_3","C_2","3C_2'","3C_2''"]
D6ct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0, -1.0,  1.0],
        [2.0,  1.0, -1.0, -2.0,  0.0,  0.0],
        [2.0, -1.0, -1.0,  2.0,  0.0,  0.0]])
