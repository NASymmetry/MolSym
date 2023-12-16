import molsym
import numpy as np
#coord_list = [[1, 2], [0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [2, 7], [2, 8], [3, 0, 4], [5, 1, 6], [7, 2, 8], [3, 0, 1], [3, 0, 2], [4, 0, 1], [4, 0, 2], [5, 1, 0], [5, 1, 2], [6, 1, 0], [6, 1, 2], [7, 2, 0], [7, 2, 1], [8, 2, 0], [8, 2, 1]]
#ic_types = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
#molsym.salcs.IC_SALCs.InternalProjectionOp("/home/smg13363/cyclopropane.xyz", coord_list, ic_types)
from molsym.salcs.internal_coordinates import InternalCoordinates
from molsym.salcs.projection_op import ProjectionOp

#coord_list = [[0,1],[0,2],[0,3],[1,0,2],[2,0,3],[3,0,1]]
#ic_types = ['R1', 'R2', 'R3', 'A1', 'A2', 'A3']
#ics = []
#for i in range(len(coord_list)):
#    ics.append([coord_list[i], ic_types[i]])

#mol = molsym.Molecule.from_file("/home/sgoodlett/MolSym/test/xyz/ammonia.xyz")
#symtext = molsym.Symtext.from_file("/home/smg13363/MolSym/test/xyz/ammonia.xyz")
#mol.symmetrize()
#symtext = molsym.Symtext.from_molecule(mol)
#ICs = InternalCoordinates(ics, symtext)
#salcs = ProjectionOp(symtext, ICs)
#print(salcs)
#print(len(salcs))
#print(salcs.basis_transformation_matrix)
np.set_printoptions(suppress=True, precision=3, linewidth=1500)
mol = molsym.Molecule.from_file("/home/sgoodlett/MolSym/test/xyz/ammonia.xyz")
#symtext = molsym.Symtext.from_file("/home/smg13363/MolSym/test/xyz/ammonia.xyz")
mol = molsym.symmetrize(mol)
symtext = molsym.Symtext.from_molecule(mol)
coords = molsym.salcs.CartesianCoordinates(symtext)
salcs = ProjectionOp(symtext, coords)

smat = salcs.basis_transformation_matrix
print(smat.T@smat)

#print(salcs.project_trans_rot())

# 30 Edges F+V-2=E
#coord_list = [[0,1],[0,4],[0,5],[0,9],[0,10],[1,2],[1,5],[1,6],[1,10],[2,3],[2,7],[2,6],[2,10],[3,4],[3,7],[3,8],[3,10],[4,8],[4,9],[4,10],[5,6],[5,9],[5,11],[6,7],[6,11],[7,8],[7,11],[8,9],[8,11],[9,11]]
#ic_types = ["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12","R13","R14","R15","R16","R17","R18","R19","R20","R21","R22","R23","R24","R25","R26","R27","R28","R29","R30"]
#molsym.salcs.IC_SALCs.InternalProjectionOp("/home/smg13363/MolSym/test/sxyz/icosahedron.xyz", coord_list, ic_types)

#coord_list = [[0,4],[0,7],[0,1],[1,3],[1,6],[2,14],[2,4],[2,3],[3,12],[4,16],[5,7],[5,10],[5,6],[6,9],[7,15],[8,10],[8,13],
#              [8,9],[9,12],[10,17],[11,12],[11,13],[11,14],[13,18],[14,19],[15,16],[15,17],[16,19],[17,18],[18,19]]
#ic_types = [f"R{i}" for i in range(30)]
#molsym.salcs.IC_SALCs.InternalProjectionOp("/home/smg13363/MolSym/test/sxyz/dodecahedron.xyz", coord_list, ic_types)

