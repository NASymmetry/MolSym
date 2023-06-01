import numpy as np
from numpy.linalg import matrix_power
from molsym.molecule import Cn, reflection_matrix, inversion_matrix, Sn
from molsym.symtext.symtext import Symel

x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

c52 = 2*np.cos(2*np.pi/5)
c54 = 2*np.cos(4*np.pi/5)

D2hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)), Symel("i",None,inversion_matrix()), 
        Symel("C_2^1",z,Cn(z,2)), Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2''(1)",y,Cn(y,2)), 
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_d(1)",x,reflection_matrix(x))]
D3hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)), 
        Symel("C_3^1",z,Cn(z,3)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,3),x.T),Cn(np.dot(Cn(z,3),x.T),2)), Symel("C_2'(3)",np.dot(matrix_power(Cn(z,3),2),x.T),Cn(np.dot(matrix_power(Cn(z,3),2),x.T),2)),
        Symel("S_3^1",z,Sn(z,3)), Symel("S_3^5",z,matrix_power(Sn(z,3),5)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",np.dot(Cn(z,3),y.T),reflection_matrix(np.dot(Cn(z,3),y.T))), Symel("sigma_v(3)",np.dot(matrix_power(Cn(z,3),2),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,3),2),y.T)))]
D4hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)), Symel("i",None,inversion_matrix()),
        Symel("C_4^1",z,Cn(z,4)),Symel("C_2^1",z,Cn(z,2)),Symel("C_4^3",z,matrix_power(Cn(z,4),3)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,4),x.T),Cn(np.dot(Cn(z,4),x.T),2)), 
        Symel("C_2''(1)",np.dot(matrix_power(Cn(z,8),1),x.T),Cn(np.dot(matrix_power(Cn(z,8),1),x.T),2)), Symel("C_2''(2)",np.dot(matrix_power(Cn(z,8),3),x.T),Cn(np.dot(matrix_power(Cn(z,8),3),x.T),2)),
        Symel("S_4^1",z,Sn(z,4)), Symel("S_4^3",z,matrix_power(Sn(z,4),3)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",x,reflection_matrix(x)), 
        Symel("sigma_d(1)",np.dot(Cn(z,8),y.T),reflection_matrix(np.dot(Cn(z,8),y.T))), Symel("sigma_d(2)",np.dot(Cn(z,8),x.T),reflection_matrix(np.dot(Cn(z,8),x.T)))]
D5hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)),
        Symel("C_5^1",z,Cn(z,5)),Symel("C_5^2",z,matrix_power(Cn(z,5),2)),Symel("C_5^3",z,matrix_power(Cn(z,5),3)), Symel("C_5^4",z,matrix_power(Cn(z,5),4)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,5),x.T),Cn(np.dot(Cn(z,5),x.T),2)), Symel("C_2'(3)",np.dot(matrix_power(Cn(z,5),2),x.T),Cn(np.dot(matrix_power(Cn(z,5),2),x.T),2)), Symel("C_2'(4)",np.dot(matrix_power(Cn(z,5),3),x.T),Cn(np.dot(matrix_power(Cn(z,5),3),x.T),2)), Symel("C_2'(5)",np.dot(matrix_power(Cn(z,5),4),x.T),Cn(np.dot(matrix_power(Cn(z,5),4),x.T),2)),
        Symel("S_5^1",z,Sn(z,5)), Symel("S_5^7",z,matrix_power(Sn(z,5),7)), Symel("S_5^3",z,matrix_power(Sn(z,5),3)), Symel("S_5^9",z,matrix_power(Sn(z,5),9)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",np.dot(Cn(z,5),y.T),reflection_matrix(np.dot(Cn(z,5),y.T))), Symel("sigma_v(3)",np.dot(matrix_power(Cn(z,5),2),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,5),2),y.T))), 
        Symel("sigma_v(4)",np.dot(matrix_power(Cn(z,5),3),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,5),3),y.T))), Symel("sigma_v(5)",np.dot(matrix_power(Cn(z,5),4),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,5),4),y.T)))]
D6hs = [Symel("E",None,np.identity(3)), Symel("sigma_h",z,reflection_matrix(z)), Symel("i",None,inversion_matrix()),
        Symel("C_6^1",z,Cn(z,6)),Symel("C_3^1",z,Cn(z,3)),Symel("C_2^1",z,Cn(z,2)),Symel("C_3^2",z,matrix_power(Cn(z,3),2)), Symel("C_6^5",z,matrix_power(Cn(z,6),5)),
        Symel("C_2'(1)",x,Cn(x,2)), Symel("C_2'(2)",np.dot(Cn(z,6),x.T),Cn(np.dot(Cn(z,6),x.T),2)), Symel("C_2'(3)",np.dot(Cn(z,3),x.T),Cn(np.dot(Cn(z,3),x.T),2)),
        Symel("C_2''(1)",np.dot(Cn(z,12),x.T),Cn(np.dot(Cn(z,12),x.T),2)), Symel("C_2''(2)",np.dot(matrix_power(Cn(z,12),3),x.T),Cn(np.dot(matrix_power(Cn(z,12),3),x.T),2)), Symel("C_2''(3)",np.dot(matrix_power(Cn(z,12),5),x.T),Cn(np.dot(matrix_power(Cn(z,12),5),x.T),2)),
        Symel("S_6^1",z,Sn(z,6)), Symel("S_3^1",z,Sn(z,3)), Symel("S_3^5",z,matrix_power(Sn(z,3),5)), Symel("S_6^5",z,matrix_power(Sn(z,6),5)),
        Symel("sigma_v(1)",y,reflection_matrix(y)), Symel("sigma_v(2)",np.dot(Cn(z,6),y.T),reflection_matrix(np.dot(Cn(z,6),y.T))), Symel("sigma_v(3)",np.dot(matrix_power(Cn(z,6),2),y.T),reflection_matrix(np.dot(matrix_power(Cn(z,6),2),y.T))),
        Symel("sigma_d(1)",np.dot(matrix_power(Cn(z,6),2),x.T),reflection_matrix(np.dot(matrix_power(Cn(z,6),2),x.T))), Symel("sigma_d(2)",x,reflection_matrix(x)), Symel("sigma_d(3)",np.dot(Cn(z,6),x.T),reflection_matrix(np.dot(Cn(z,6),x.T)))]

D2hirr = ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"]
D2hcn = ["E","C_2(z)","C_2(y)","C_2(x)","i","sigma(xy)","sigma(xz)","sigma(yz)"]
D2hct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],
        [1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0],
        [1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0]])

D3hirr = ["A1'", "A2'", "E'", "A1''", "A2''", "E''"] 
D3hcn = ["E","2C_3","3C_2","sigma_h","2S_3","3sigma_v"]
D3hct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        [2.0, -1.0,  0.0,  2.0, -1.0,  0.0],
        [1.0,  1.0,  1.0, -1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0, -1.0, -1.0,  1.0],
        [2.0, -1.0,  0.0, -2.0,  1.0,  0.0]])

D4hirr = ["A1g", "A2g", "B1g", "B2g", "Eg", "A1u", "A2u", "B1u", "B2u", "Eu"]
D4hcn = ["E","2C_4","C_2","2C_2'","2C_2''","i","2S_4","sigma_h","2sigma_v","2sigma_d"]
D4hct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [2.0,  0.0, -2.0,  0.0,  0.0,  2.0,  0.0, -2.0,  0.0,  0.0],
        [1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],
        [1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [2.0,  0.0, -2.0,  0.0,  0.0, -2.0,  0.0,  2.0,  0.0,  0.0]])

D5hirr = ["A1'", "A2'", "E1'", "E2'", "A1''", "A2''", "E1''", "E2''"]
D5hcn = ["E","2C_5","2C_5^2","5C_2","sigma_h","2S_5","2S_5^3","5sigma_v"]
D5hct = np.array([[1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0, 1.0, -1.0,  1.0,  1.0,  1.0, -1.0],
        [2.0,  c52, c54,  0.0,  2.0,  c52,  c54,  0.0],
        [2.0,  c54, c52,  0.0,  2.0,  c54,  c52,  0.0],
        [1.0,  1.0, 1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0,  1.0, 1.0, -1.0, -1.0, -1.0, -1.0,  1.0],
        [2.0,  c52, c54,  0.0, -2.0, -c52, -c54,  0.0],
        [2.0,  c54, c52,  0.0, -2.0, -c54, -c52,  0.0]])

D6hirr = ["A1g", "A2g", "B1g", "B2g", "E1g", "E2g", "A1u", "A2u", "B1u", "B2u", "E1u", "E2u"]
D6hcn = ["E","2C_6","2C_3","C_2","3C_2'","3C_2''","i","2S_3","2S_6","sigma_h","3sigma_d","3sigma_v"]
D6hct = np.array([[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0],
        [2.0,  1.0, -1.0, -2.0,  0.0,  0.0,  2.0,  1.0, -1.0, -2.0,  0.0,  0.0],
        [2.0, -1.0, -1.0,  2.0,  0.0,  0.0,  2.0, -1.0, -1.0,  2.0,  0.0,  0.0],
        [1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],
        [1.0, -1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        [2.0,  1.0, -1.0, -2.0,  0.0,  0.0, -2.0, -1.0,  1.0,  2.0,  0.0,  0.0],
        [2.0, -1.0, -1.0,  2.0,  0.0,  0.0, -2.0,  1.0,  1.0, -2.0,  0.0,  0.0]])