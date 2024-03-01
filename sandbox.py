import molsym
import numpy as np
from molsym.symtext.multiplication_table import build_mult_table, subgroups, subgroups_better, subgroup_by_name
from molsym.salcs.internal_coordinates import InternalCoordinates
from molsym.salcs.spherical_harmonics import SphericalHarmonics
from molsym.salcs.projection_op import ProjectionOp
import timeit

mol = molsym.Molecule.from_file("/home/smg13363/MolSym/nate.xyz")
mol = molsym.symmetrize(mol, asym_tol=0.05)
symtext = molsym.Symtext.from_molecule(mol)
print(symtext.pg)
print(mol)

#start_time = timeit.default_timer()
#symels = molsym.symtext.main.pg_to_symels("D2h")
#print(symels)
#mult_table = build_mult_table(symels)
#print(timeit.default_timer()-start_time)
#print(mult_table)

#start_time = timeit.default_timer()
#sg1 = subgroups(symels, mult_table)
#print(timeit.default_timer()-start_time)

#sg2 = subgroups_better(symels, mult_table)
#print(timeit.default_timer()-start_time)
#start_time = timeit.default_timer()
#print(subgroup_by_name(symels, mult_table, "C2v"))
#print(timeit.default_timer()-start_time)

#mol = molsym.Molecule.from_file("/home/smg13363/MolSym/test/xyz/benzene.xyz")
#mol = molsym.symmetrize(mol)
#print(mol.find_SEAs())
#symtext = molsym.Symtext.from_molecule(mol)
#print(symtext.chartable)
#subgroup_symtext = symtext.subgroup_symtext("D2h")
#print(subgroup_symtext)

#symm_equiv = []
#done = []
#for atom_i in range(len(subgroup_symtext.mol)):
#    if atom_i in done:
#        continue
#    equiv_set = []
#    for sidx in range(len(subgroup_symtext)):
#        newatom = subgroup_symtext.atom_map[atom_i, sidx]
#        equiv_set += [newatom]
#    reduced_equiv_set = list(set(equiv_set))
#    symm_equiv.append(reduced_equiv_set)
#    done += reduced_equiv_set
#print(symm_equiv)
#print(subgroup_symtext.complex)

#coord_list = [[0,1],[0,2],[0,3],[1,0,2],[2,0,3],[3,0,1]]
#ic_types = ['R1', 'R2', 'R3', 'A1', 'A2', 'A3']
#coord_list = [[0,1],[0,2],[1,0,2]]
#ic_types = ['R1', 'R2', 'A1']
#coord_list = [[0,1],[0,2],[0,3],[0,4],[1,5],[2,0,1],[3,0,1],[4,0,1],[0,1,5],[2,0,1,5],[3,0,1,5],[4,0,1,5]]
#ic_types = ['R1','R2','R3','R4','R5','A1','A2','A3','A4','D1','D2','D3']
#fn = "/home/smg13363/MolSym/test/xyz/methane.xyz"
#coord_list = [[0,1],[0,2],[0,3],[0,4],[1,0,2],[1,0,3],[1,0,4],[2,0,3],[2,0,4],[3,0,4]]
#ic_types = ['R1', 'R2', 'R3', 'R4', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
#fn = "/home/smg13363/MolSym/test/xyz/methane.xyz"
#coord_list = [[0,1],[1,2],[2,3],[3,4],[4,5],[0,5],[0,6],[1,7],[2,8],[3,9],[4,10],[5,11],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],[0,1,7],[1,2,8],[2,3,9],[3,4,10],[4,5,11],[5,0,6],[0,5,11],[5,4,10],[4,3,9],[3,2,8],[2,1,7],[1,0,6]]
#ic_types = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18']
#ics = []
#for i in range(len(coord_list)):
#    ics.append([coord_list[i], ic_types[i]])
#ICs = InternalCoordinates(symtext, ics)
#salcs = ProjectionOp(symtext, ICs)
#print(salcs)
#np.set_printoptions(suppress=True, precision=2, linewidth=1500)

#mol = molsym.Molecule.from_file(fn)
#mol = molsym.symmetrize(mol)
#symtext = molsym.Symtext.from_molecule(mol)
#print(symtext.mol)
#ICs = InternalCoordinates(symtext, ics)
#salcs = ProjectionOp(symtext, ICs)
#print(salcs)
#print(len(salcs))
#print(salcs.basis_transformation_matrix)
#np.set_printoptions(suppress=True, precision=3, linewidth=1500)
#mol = molsym.Molecule.from_file("/home/sgoodlett/MolSym/test/xyz/ammonia.xyz")
##symtext = molsym.Symtext.from_file("/home/smg13363/MolSym/test/xyz/ammonia.xyz")
#mol = molsym.symmetrize(mol)
#symtext = molsym.Symtext.from_molecule(mol)
#coords = molsym.salcs.CartesianCoordinates(symtext)
#salcs = ProjectionOp(symtext, coords)
#
#smat = salcs.basis_transformation_matrix
#print(smat.T@smat)
#
##print(salcs.project_trans_rot())
#
## 30 Edges F+V-2=E
##coord_list = [[0,1],[0,4],[0,5],[0,9],[0,10],[1,2],[1,5],[1,6],[1,10],[2,3],[2,7],[2,6],[2,10],[3,4],[3,7],[3,8],[3,10],[4,8],[4,9],[4,10],[5,6],[5,9],[5,11],[6,7],[6,11],[7,8],[7,11],[8,9],[8,11],[9,11]]
##ic_types = ["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12","R13","R14","R15","R16","R17","R18","R19","R20","R21","R22","R23","R24","R25","R26","R27","R28","R29","R30"]
##molsym.salcs.IC_SALCs.InternalProjectionOp("/home/smg13363/MolSym/test/sxyz/icosahedron.xyz", coord_list, ic_types)
#
##coord_list = [[0,4],[0,7],[0,1],[1,3],[1,6],[2,14],[2,4],[2,3],[3,12],[4,16],[5,7],[5,10],[5,6],[6,9],[7,15],[8,10],[8,13],
##              [8,9],[9,12],[10,17],[11,12],[11,13],[11,14],[13,18],[14,19],[15,16],[15,17],[16,19],[17,18],[18,19]]
##ic_types = [f"R{i}" for i in range(30)]
##molsym.salcs.IC_SALCs.InternalProjectionOp("/home/smg13363/MolSym/test/sxyz/dodecahedron.xyz", coord_list, ic_types)
#
#

# Spherical harmonic salcs
#mol = molsym.Molecule.from_file("/home/smg13363/MolSym/test/xyz/benzene.xyz")
#mol = molsym.symmetrize(mol)
#symtext = molsym.Symtext.from_molecule(mol)
#symtext = symtext.subgroup_symtext("C3")
##symtext = symtext.largest_D2h_subgroup()
#print(symtext)
#bset = [[0,0,0,0,1,1,1,2,2,3],[0,0,0,1,1,2],[0,0,0,1,1,2],[0,0,0,1,1,2]]
#cbs = [0,0,1]
#hbs = [0,1]
#bset = sum([[cbs for i in range(6)],[hbs for i in range(6)]],[])
#coords = SphericalHarmonics(symtext, bset)
#salcs = ProjectionOp(symtext, coords)
#np.set_printoptions(suppress=True, precision=3, linewidth=1500, threshold=np.inf)
## nb on each C: 14
#nbfpa = 5
#cidx = 3
#select = [cidx +  nbfpa*i for i in range(6)] + [cidx+1 +  nbfpa*i for i in range(6)]
#print([len(salcs.salcs_by_irrep[i]) for i in range(len(salcs.irreps))])
#print(np.linalg.matrix_rank(salcs.basis_transformation_matrix))
#print(np.max(np.imag(salcs.basis_transformation_matrix)))
#print(salcs.basis_transformation_matrix.dtype)
#ctr = 0
#for i in range(len(salcs.irreps)):
#    n = len(salcs.salcs_by_irrep[i])
#    print(salcs.irreps[i])
#    sm = salcs.basis_transformation_matrix.T[ctr:ctr+n,select]
#    nt = np.isclose(sm, 0).all(axis=1)
#    print(sm[~nt,:])
#    ctr += n
#td = molsym.symtext.irrep_mats.irrm_Ih
#e = td["T1g"]
#t1 = td["Hg"]
#dim1 = 3
#dim2 = 5
#a = np.zeros((dim1,dim1,dim2,dim2,dim1,dim1,dim2,dim2))
#for i in range(dim1):
#    for ib in range(dim1):
#        for k in range(dim1):
#            for kb in range(dim1):
#                for j in range(dim2):
#                    for jb in range(dim2):
#                        for l in range(dim2):
#                            for lb in range(dim2):
#                                a[i,k,j,l,ib,kb,jb,lb] += np.sum(e[:,i,ib]*e[:,k,kb]*t1[:,j,jb]*t1[:,l,lb])
#b = np.random.rand(dim1,dim1,dim2,dim2)
#c = np.einsum("ijklmnop,mnop->ijkl", a, b)
#print(c.reshape((9,25)))





