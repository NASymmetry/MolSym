import numpy as np
from numpy.linalg import matrix_power
from molsym.molecule import Cn, reflection_matrix, inversion_matrix, Sn
from molsym.symtext.symtext import Symel

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

rt2 = np.sqrt(2)
rt3 = np.sqrt(3)
c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

D2ds = [Symel("E",None,np.identity(3)), Symel("C_2^1",z,Cn(z,2)),
        Symel("S_4^1",z,Sn(z,4)), Symel("S_4^3",z,matrix_power(Sn(z,4),3)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2''(1)",y,Cn(y,2)),
        Symel("sigma_d(1)",np.dot(Cn(z,8),y.T),reflection_matrix(np.dot(Cn(z,8),y.T))), Symel("sigma_d(2)",np.dot(Cn(z,8),x.T),reflection_matrix(np.dot(Cn(z,8),x.T)))]
D3ds = [Symel("E",None,np.identity(3)), Symel("i",None,inversion_matrix()), 
        Symel("C_3^1",z,Cn(z,3)), Symel("C_3^2",z,matrix_power(Cn(z,3),2)),
        Symel("S_6^1",z,Sn(z,6)), Symel("S_6^5",z,matrix_power(Sn(z,6),5)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,3),x.T),Cn(np.dot(Cn(z,3),x.T),2)), Symel("C_2'(3)",np.dot(matrix_power(Cn(z,3),2),x.T),Cn(np.dot(matrix_power(Cn(z,3),2),x.T),2)),
        Symel("sigma_d(1)",np.dot(Cn(z,12),y.T),reflection_matrix(np.dot(Cn(z,12),y.T))), Symel("sigma_d(2)",x,reflection_matrix(x)), Symel("sigma_d(3)",np.dot(matrix_power(Cn(z,12),5),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,12),5),y.T)))]
D4ds = [Symel("E",None,np.identity(3)),
        Symel("C_4^1",z,Cn(z,4)),Symel("C_2^1",z,Cn(z,2)),Symel("C_4^3",z,matrix_power(Cn(z,4),3)),
        Symel("S_8^1",z,Sn(z,8)), Symel("S_8^3",z,matrix_power(Sn(z,8),3)), Symel("S_8^5",z,matrix_power(Sn(z,8),5)), Symel("S_8^7",z,matrix_power(Sn(z,8),7)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,4),x.T),Cn(np.dot(Cn(z,4),x.T),2)), 
        Symel("C_2''(1)",np.dot(matrix_power(Cn(z,8),1),x.T),Cn(np.dot(matrix_power(Cn(z,8),1),x.T),2)), Symel("C_2''(2)",np.dot(matrix_power(Cn(z,8),3),x.T),Cn(np.dot(matrix_power(Cn(z,8),3),x.T),2)),
        Symel("sigma_d(1)",np.dot(Cn(z,16),y.T),reflection_matrix(np.dot(Cn(z,16),y.T))), Symel("sigma_d(2)",np.dot(matrix_power(Cn(z,16),3),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,16),3),y.T))), 
        Symel("sigma_d(3)",np.dot(matrix_power(Cn(z,16),5),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,16),5),y.T))), Symel("sigma_d(4)",np.dot(matrix_power(Cn(z,16),7),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,16),7),y.T)))]
D5ds = [Symel("E",None,np.identity(3)), Symel("i",None,inversion_matrix()),
        Symel("C_5^1",z,Cn(z,5)),Symel("C_5^2",z,matrix_power(Cn(z,5),2)),Symel("C_5^3",z,matrix_power(Cn(z,5),3)), Symel("C_5^4",z,matrix_power(Cn(z,5),4)),
        Symel("S_10^1",z,Sn(z,10)), Symel("S_10^3",z,matrix_power(Sn(z,10),3)), Symel("S_10^7",z,matrix_power(Sn(z,10),7)), Symel("S_10^9",z,matrix_power(Sn(z,10),9)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,5),x.T),Cn(np.dot(Cn(z,5),x.T),2)), Symel("C_2'(3)",np.dot(matrix_power(Cn(z,5),2),x.T),Cn(np.dot(matrix_power(Cn(z,5),2),x.T),2)), Symel("C_2'(4)",np.dot(matrix_power(Cn(z,5),3),x.T),Cn(np.dot(matrix_power(Cn(z,5),3),x.T),2)), Symel("C_2'(5)",np.dot(matrix_power(Cn(z,5),4),x.T),Cn(np.dot(matrix_power(Cn(z,5),4),x.T),2)),
        Symel("sigma_d(1)",np.dot(Cn(z,20),y.T),reflection_matrix(np.dot(Cn(z,20),y.T))), 
        Symel("sigma_d(2)",np.dot(matrix_power(Cn(z,20),3),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,20),3),y.T))),
        Symel("sigma_d(3)",np.dot(matrix_power(Cn(z,20),5),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,20),5),y.T))),
        Symel("sigma_d(4)",np.dot(matrix_power(Cn(z,20),7),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,20),7),y.T))),
        Symel("sigma_d(5)",np.dot(matrix_power(Cn(z,20),9),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,20),9),y.T)))]
D6ds = [Symel("E",None,np.identity(3)),
        Symel("C_6^1",z,Cn(z,6)),Symel("C_3^1",z,Cn(z,3)),Symel("C_2^1",z,Cn(z,2)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)), Symel("C_6^5",z,matrix_power(Cn(z,6),5)),
        Symel("S_12^1",z,Sn(z,12)),
        Symel("S_4^1",z,Sn(z,4)),
        Symel("S_12^5",z,matrix_power(Sn(z,12),5)),
        Symel("S_12^7",z,matrix_power(Sn(z,12),7)),
        Symel("S_4^3",z,matrix_power(Sn(z,4),3)),
        Symel("S_12^11",z,matrix_power(Sn(z,12),11)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,6),x.T),Cn(np.dot(Cn(z,6),x.T),2)), Symel("C_2'(3)",np.dot(Cn(z,3),x.T),Cn(np.dot(Cn(z,3),x.T),2)),
        Symel("C_2''(1)",np.dot(Cn(z,12),x.T),Cn(np.dot(Cn(z,12),x.T),2)), Symel("C_2''(2)",np.dot(matrix_power(Cn(z,12),3),x.T),Cn(np.dot(matrix_power(Cn(z,12),3),x.T),2)), Symel("C_2''(3)",np.dot(matrix_power(Cn(z,12),5),x.T),Cn(np.dot(matrix_power(Cn(z,12),5),x.T),2)),
        Symel("sigma_d(1)",np.dot(matrix_power(Cn(z,24),1),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,24),1),y.T))),
        Symel("sigma_d(2)",np.dot(matrix_power(Cn(z,24),3),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,24),3),y.T))),
        Symel("sigma_d(3)",np.dot(matrix_power(Cn(z,24),5),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,24),5),y.T))),
        Symel("sigma_d(4)",np.dot(matrix_power(Cn(z,24),7),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,24),7),y.T))),
        Symel("sigma_d(5)",np.dot(matrix_power(Cn(z,24),9),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,24),9),y.T))),
        Symel("sigma_d(6)",np.dot(matrix_power(Cn(z,24),11),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,24),11),y.T)))]


D2dirr = ["A1", "A2", "B1", "B2", "E"]
D2dcn = ["E","2S_4","C_2","2C_2'","2sigma_d"]
D2dct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0,  1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0],
        [2.0,  0.0, -2.0,  0.0,  0.0]])

D3dirr = ["A1g", "A2g", "Eg", "A1u", "A2u", "Eu"]
D3dcn = ["E","2C_3","3C_2","i","2S_6","3sigma_d"]
D3dct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        [2.0, -1.0,  0.0,  2.0, -1.0,  0.0],
        [1.0,  1.0,  1.0, -1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0, -1.0, -1.0,  1.0],
        [2.0, -1.0,  0.0, -2.0,  1.0,  0.0]])

D4dirr = ["A1", "A2", "B1", "B2", "E1", "E2", "E3"]
D4dcn = ["E","2S_8","2C_4","2S_8^3","C_2","4C_2'","4sigma_d"]
D4dct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [2.0,  rt2,  0.0, -rt2, -2.0,  0.0,  0.0],
        [2.0,  0.0, -2.0,  0.0,  2.0,  0.0,  0.0],
        [2.0, -rt2,  0.0,  rt2, -2.0,  0.0,  0.0]])

D5dirr = ["A1g", "A2g", "E1g", "E2g", "A1u", "A2u", "E1u", "E2u"]
D5dcn = ["E","2C_5","2C_5^2","5C_2","i","2S_10^3","2S_10","5sigma_d"]
D5dct = np.array([[1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0, 1.0, -1.0,  1.0,  1.0,  1.0, -1.0],
        [2.0,  c52, c54,  0.0,  2.0,  c52,  c54,  0.0],
        [2.0,  c54, c52,  0.0,  2.0,  c54,  c52,  0.0],
        [1.0,  1.0, 1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0,  1.0, 1.0, -1.0, -1.0, -1.0, -1.0,  1.0],
        [2.0,  c52, c54,  0.0, -2.0, -c52, -c54,  0.0],
        [2.0,  c54, c52,  0.0, -2.0, -c54, -c52,  0.0]])

D6dirr = ["A1", "A2", "B1", "B2", "E1", "E2", "E3", "E4", "E5"]
D6dcn = ["E","2S_12","2C_6","2S_4","2C_3","2S_12^5","C_2","6C_2'","6sigma_d"]
D6dct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [2.0,  rt3,  1.0,  0.0, -1.0, -rt3, -2.0,  0.0,  0.0],
        [2.0,  1.0, -1.0, -2.0, -1.0,  1.0,  2.0,  0.0,  0.0],
        [2.0,  0.0, -2.0,  0.0,  2.0,  0.0, -2.0,  0.0,  0.0],
        [2.0, -1.0, -1.0,  2.0, -1.0, -1.0,  2.0,  0.0,  0.0],
        [2.0, -rt3,  1.0,  0.0, -1.0,  rt3, -2.0,  0.0,  0.0]])