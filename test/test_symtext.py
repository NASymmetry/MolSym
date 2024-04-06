import pytest
import numpy as np
import molsym
from molsym.symtext.symel import pg_to_symels
from molsym.symtext.character_table import pg_to_chartab, grab_class_orders
from molsym.symtext.symtext_helper import generate_symel_to_class_map, rotate_mol_to_symels
from molsym.symtext.multiplication_table import build_mult_table

# C1, Ci, Cs, C2v, C3h, S8, D6h, Td, Oh, Ih
classes_test_set = [
    ['E'],
    ['E', 'i'],
    ['E', 'sigma_h'],
    ['E', 'C_2', 'sigma_v(xz)', 'sigma_d(yz)'],
    ['E', 'C_3', 'C_3^2', 'sigma_h', 'S_3', 'S_3^5'],
    ['E', 'S_8', 'C_4', 'S_8^3', 'C_2', 'S_8^5', 'C_4^3', 'S_8^7'],
    ['E', '2C_6', '2C_3', 'C_2', "3C_2'", "3C_2''", 'i', '2S_3', '2S_6', 'sigma_h', '3sigma_d', '3sigma_v'],
    ['E', '8C_3', '3C_2', '6S_4', '6sigma_d'],
    ['E', '8C_3', '6C_2', '6C_4', '3C_2', 'i', '6S_4', '8S_6', '3sigma_h', '6sigma_d'],
    ['E', '12C_5', '12C_5^2', '20C_3', '15C_2', 'i', '12S_10', '12S_10^3', '20S_6', '15sigma_']]
class_orders_test_set = [
    [1],
    [1, 1],
    [1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3],
    [1, 8, 3, 6, 6],
    [1, 8, 6, 6, 3, 1, 6, 8, 3, 6],
    [1, 12, 12, 20, 15, 1, 12, 12, 20, 15]]

@pytest.mark.parametrize("class_list, class_orders", [(classes_test_set[i], class_orders_test_set[i]) for i in range(len(classes_test_set))])
def test_grab_class_orders(class_list, class_orders):
    assert (grab_class_orders(class_list) == class_orders).all()

pgs = ["C1", "Ci", "Cs", "C2v", "C3h", "S8", "D6h", "Td", "Oh", "Ih"]
symel_to_class_map_test_set = [
    [0],
    [0, 1],
    [0, 1],
    [0, 1, 2, 3],
    [0, 3, 1, 2, 4, 5],
    [0, 2, 4, 6, 1, 3, 5, 7],
    [0, 9, 6, 1, 2, 3, 2, 1, 4, 4, 4, 5, 5, 5, 8, 7, 7, 8, 11, 11, 11, 10, 10, 10],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3],
    [0, 3, 4, 3, 3, 4, 3, 3, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 5, 6, 8, 6, 6, 8, 6, 6, 8, 6, 7, 7, 7, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9],
    [0, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
     5, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6,
     8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]

@pytest.mark.parametrize("pg, answer", [(pgs[i], symel_to_class_map_test_set[i]) for i in range(len(pgs))])
def test_generate_symel_to_class_map(pg, answer):
    symels = pg_to_symels(pg)
    ctab = pg_to_chartab(pg)
    mp = generate_symel_to_class_map(symels, ctab)
    print(pg)
    print(mp)
    assert (mp == answer).all()

pgs = ["C1", "Ci", "Cs", "C2v", "C3h", "S8", "D3h"]
mult_table_test_set = [
    [0],
    [[0,1],[1,0]],
    [[0,1],[1,0]],
    [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]],
    [[0,1,2,3,4,5],[1,0,4,5,2,3],[2,4,3,0,5,1],[3,5,0,2,1,4],[4,2,5,1,3,0],[5,3,1,4,0,2]],
    [[0,1,2,3,4,5,6,7],[1,2,3,0,5,6,7,4],[2,3,0,1,6,7,4,5],[3,0,1,2,7,4,5,6],[4,5,6,7,1,2,3,0],[5,6,7,4,2,3,0,1],[6,7,4,5,3,0,1,2],[7,4,5,6,0,1,2,3]],
    [   [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11],
        [ 1, 0, 7, 8, 9,10,11, 2, 3, 4, 5, 6],
        [ 2, 7, 3, 0, 6, 4, 5, 8, 1,11, 9,10],
        [ 3, 8, 0, 2, 5, 6, 4, 1, 7,10,11, 9],
        [ 4, 9, 5, 6, 0, 2, 3,10,11, 1, 7, 8],
        [ 5,10, 6, 4, 3, 0, 2,11, 9, 8, 1, 7],
        [ 6,11, 4, 5, 2, 3, 0, 9,10, 7, 8, 1],
        [ 7, 2, 8, 1,11, 9,10, 3, 0, 6, 4, 5],
        [ 8, 3, 1, 7,10,11, 9, 0, 2, 5, 6, 4],
        [ 9, 4,10,11, 1, 7, 8, 5, 6, 0, 2, 3],
        [10, 5,11, 9, 8, 1, 7, 6, 4, 3, 0, 2],
        [11, 6, 9,10, 7, 8, 1, 4, 5, 2, 3, 0]]]

@pytest.mark.parametrize("pg, answer", [(pgs[i], mult_table_test_set[i]) for i in range(len(pgs))])
def test_build_mult_table(pg, answer):
    symels = pg_to_symels(pg)
    mult_table = build_mult_table(symels)
    assert (mult_table == answer).all()

axes_test_set = [
    (np.array([1,0,0]), np.array([0,1,0])), # All permutations of identity
    (np.array([1,0,0]), np.array([0,0,1])),
    (np.array([0,1,0]), np.array([1,0,0])),
    (np.array([0,1,0]), np.array([0,0,1])),
    (np.array([0,0,1]), np.array([1,0,0])),
    (np.array([0,0,1]), np.array([0,1,0])),
    (np.array([0,0,0]), np.array([0,0,0])), # No axis
    (np.array([0,0,1]), np.array([0,0,0])), # No saxis
    (np.array([1,0,0]), np.array([0,0,0])),
    (np.array([0.54589206, 0.4508254 , 0.70622823]), np.array([0,0,0])), # Random normalized vectors, paxis and saxis are orthogonal
    (np.array([-0.26261511, -0.53822585,  0.80084096]), np.array([0.75128557, 0.40675235, 0.51973312])),
    (np.array([0.27547511, 0.90122838, 0.33451588]), np.array([0.66963905,  0.06976378, -0.73940284]))]

rotation_matrices = [
    np.array([[0,1,0],[0,0,1],[1,0,0]]).T,
    np.array([[0,0,1],[0,-1,0],[1,0,0]]).T,
    np.array([[1,0,0],[0,0,-1],[0,1,0]]).T,
    np.array([[0,0,1],[1,0,0],[0,1,0]]).T,
    np.array([[1,0,0],[0,1,0],[0,0,1]]).T,
    np.array([[0,1,0],[-1,0,0],[0,0,1]]).T,
    np.array([[1,0,0],[0,1,0],[0,0,1]]),
    np.array([[0,1,0],[-1,0,0],[0,0,1]]),
    np.array([[0,0,-1],[0,1,0],[1,0,0]]).T,
    np.array([[ 0.        , -0.84289979,  0.53807058],[ 0.83785551, -0.29372846, -0.4601323 ],[0.54589206, 0.4508254 , 0.70622823]]).T,
    np.array([[0.75128557, 0.40675235, 0.51973312],[-0.60547774,  0.73815003,  0.297542  ],[-0.26261511, -0.53822585,  0.80084096]]).T,
    np.array([[0.66963905,  0.06976378, -0.73940284],[-0.68970791,  0.42769197, -0.58427953],[0.27547511, 0.90122838, 0.33451588]]).T
]

@pytest.mark.parametrize("paxis, saxis, answer", [(axes_test_set[i][0], axes_test_set[i][1], rotation_matrices[i]) for i in range(len(axes_test_set))])
def test_rotate_mol_to_symels(paxis, saxis, answer):
    mol = molsym.Molecule.from_file("test/xyz/water.xyz")
    new_mol, rmat, rmat_inv = rotate_mol_to_symels(mol, paxis, saxis)
    print(rmat)
    print(answer)
    assert np.isclose(rmat, answer).all()
    assert np.isclose(rmat_inv, answer.T).all()
    
# Water, symmetrized ammonia, methane, benzene
fns = ["C1", "water", "ammonia", "methane", "benzene"]
pgs = ["C1", "C2v", "C3v", "Td", "D6h"]
mol_test_set = [(["H", "C", "N", "O"], np.array([[-0.52729414,-0.61531157,-0.70283540],[1.36243199,-0.61531157,-0.70283540],
                                                  [-0.52729414, 1.27441455,-0.70283540],[-0.52729414,-0.61531157, 1.18689073]])),
                (['O', 'H', 'H'], np.array([[0, 0, 0.128200553],[0, -1.47972489, -1.01731791],[0,  1.47972489, -1.01731791]])),
                (['N', 'H', 'H', 'H'], np.array([[0, 0, 0.131258858],[-0.881225646, -1.52632759, -0.607918852],
                                                 [-0.881225646,  1.52632759, -0.607918852],[ 1.76245129, 0, -0.607918852]])),
                (['C', 'H', 'H', 'H', 'H'], np.array([[0, 0, 0],[ 1.18465230, -1.18465230,  1.18465230],[-1.18465230,  1.18465230,  1.18465230],
                                                      [ 1.18465230,  1.18465230, -1.18465230],[-1.18465230, -1.18465230, -1.18465230]])),
                (['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], 
                 np.array([[ 2.62988083e+00,  0.00000000e+00,  0.00000000e+00],[ 1.31494041, -2.27754361, 0],[-1.31494042, -2.27754361, 0],
                           [-2.62988083, 0, 0],[-1.31494041,  2.27754361, 0],[ 1.31494042,  2.27754361, 0],[ 4.66684496, 0, 0],
                           [ 2.33342248, -4.04160629, 0],[-2.33342248, -4.04160629, 0],[-4.66684496, 0, 0],[-2.33342248,  4.04160629, 0],
                           [ 2.33342248,  4.04160629, 0]]))]

atom_map_test_set = [[[0],[1],[2],[3]],
                     [[0,0,0,0],[1,2,2,1],[2,1,1,2]],
                     [[0,0,0,0,0,0],[1,3,2,2,3,1],[2,1,3,1,2,3],[3,2,1,3,1,2]],
                     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 3, 2, 1, 1, 2, 4, 4, 3, 3, 4, 2, 1, 2, 4, 1, 1, 3, 4, 2, 3, 2, 3, 4],
                      [2, 1, 3, 3, 4, 4, 1, 2, 2, 4, 3, 1, 2, 1, 2, 3, 4, 2, 1, 3, 1, 4, 4, 3],
                      [3, 2, 1, 4, 2, 3, 3, 1, 4, 1, 2, 4, 4, 3, 3, 2, 3, 1, 2, 4, 4, 1, 2, 1],
                      [4, 4, 4, 2, 3, 1, 2, 3, 1, 2, 1, 3, 3, 4, 1, 4, 2, 4, 3, 1, 2, 3, 1, 2]],
                     [[ 0,  0,  3,  5,  4,  3,  2,  1,  0,  4,  2,  5,  3,  1,  5,  4,  2,  1,  0,  4,  2,  5,  3,  1],
                      [ 1,  1,  4,  0,  5,  4,  3,  2,  5,  3,  1,  4,  2,  0,  0,  5,  3,  2,  5,  3,  1,  4,  2,  0],
                      [ 2,  2,  5,  1,  0,  5,  4,  3,  4,  2,  0,  3,  1,  5,  1,  0,  4,  3,  4,  2,  0,  3,  1,  5],
                      [ 3,  3,  0,  2,  1,  0,  5,  4,  3,  1,  5,  2,  0,  4,  2,  1,  5,  4,  3,  1,  5,  2,  0,  4],
                      [ 4,  4,  1,  3,  2,  1,  0,  5,  2,  0,  4,  1,  5,  3,  3,  2,  0,  5,  2,  0,  4,  1,  5,  3],
                      [ 5,  5,  2,  4,  3,  2,  1,  0,  1,  5,  3,  0,  4,  2,  4,  3,  1,  0,  1,  5,  3,  0,  4,  2],
                      [ 6,  6,  9, 11, 10,  9,  8,  7,  6, 10,  8, 11,  9,  7, 11, 10,  8,  7,  6, 10,  8, 11,  9,  7],
                      [ 7,  7, 10,  6, 11, 10,  9,  8, 11,  9,  7, 10,  8,  6,  6, 11,  9,  8, 11,  9,  7, 10,  8,  6],
                      [ 8,  8, 11,  7,  6, 11, 10,  9, 10,  8,  6,  9,  7, 11,  7,  6, 10,  9, 10,  8,  6,  9,  7, 11],
                      [ 9,  9,  6,  8,  7,  6, 11, 10,  9,  7, 11,  8,  6, 10,  8,  7, 11, 10,  9,  7, 11,  8,  6, 10],
                      [10, 10,  7,  9,  8,  7,  6, 11,  8,  6, 10,  7, 11,  9,  9,  8,  6, 11,  8,  6, 10,  7, 11,  9],
                      [11, 11,  8, 10,  9,  8,  7,  6,  7, 11,  9,  6, 10,  8, 10,  9,  7,  6,  7, 11,  9,  6, 10,  8]]]

complex_test_set = [False, False, False, False, False]
order_test_set = [1,4,6,24,24]

@pytest.mark.parametrize("i", [i for i in range(len(fns))])
def test_Symtext(i):
    angstrom_per_bohr = 0.529177249
    mol = molsym.Molecule.from_file("test/xyz/"+fns[i]+".xyz")
    mol = molsym.symmetrize(mol)
    symtext = molsym.Symtext.from_molecule(mol)
    # Add mult table, symels, ctab, class map?
    assert (mol_test_set[i][0] == mol.atoms).all()
    assert np.isclose(mol_test_set[i][1]*angstrom_per_bohr, mol.coords).all() # QCElemental performed undesired unit conv. in test set
    assert symtext.pg.str == pgs[i]
    assert (symtext.atom_map == atom_map_test_set[i]).all()
    assert complex_test_set[i] == symtext.complex
    assert order_test_set[i] == symtext.order

fns_D2h_subgroups = ["C1", "water", "ammonia", "methane", "benzene"]
D2h_subgroup_pgs = ["C1", "C2v", "Cs", "D2", "D2h"]
D2h_subgroup_atom_map_test_set = [
    [[0],[1],[2],[3]],
    [[0,0,0,0],[1,2,1,2],[2,1,2,1]],
    [[0,0],[1,2],[2,1],[3,3]],
    [[0,0,0,0],[1,3,2,4],[2,4,1,3],[3,1,4,2],[4,2,3,1]],
    [[0,0,3,3,3,0,3,0],[1,1,4,4,2,5,2,5],[2,2,5,5,1,4,1,4],[3,3,0,0,0,3,0,3],[4,4,1,1,5,2,5,2],[5,5,2,2,4,1,4,1],
     [6,6,9,9,9,6,9,6],[7,7,10,10,8,11,8,11],[8,8,11,11,7,10,7,10],[9,9,6,6,6,9,6,9],[10,10,7,7,11,8,11,8],[11,11,8,8,10,7,10,7]]
]
D2h_subgroup_complex_test_set = [False, False, False, False, False]
D2h_subgroup_order_test_set = [1,4,2,4,8]

@pytest.mark.parametrize("i", [i for i in range(len(fns_D2h_subgroups))])
def test_Symtext_largest_D2h_subgroup(i):
    mol = molsym.Molecule.from_file("test/xyz/"+fns_D2h_subgroups[i]+".xyz")
    mol = molsym.symmetrize(mol)
    symtext = molsym.Symtext.from_molecule(mol)
    symtext = symtext.largest_D2h_subgroup()
    #assert (mol_test_set[i][0] == mol.atoms).all()
    #assert np.isclose(mol_test_set[i][1], mol.coords).all()
    assert symtext.pg.str == D2h_subgroup_pgs[i]
    assert (symtext.atom_map == D2h_subgroup_atom_map_test_set[i]).all()
    assert D2h_subgroup_complex_test_set[i] == symtext.complex
    assert D2h_subgroup_order_test_set[i] == symtext.order

subgroup_fns = ["water", "benzene", "benzene", "benzene"]
subgroup_pgs = ["C2", "D3h", "C3", "C4"]
subgroup_atom_map_test_set = [
    [[0,0],[1,2],[2,1]],
    [[ 0,  0,  4,  2,  2,  4,  0,  4,  2,  2,  4,  0],
     [ 1,  1,  5,  3,  1,  3,  5,  5,  3,  1,  3,  5],
     [ 2,  2,  0,  4,  0,  2,  4,  0,  4,  0,  2,  4],
     [ 3,  3,  1,  5,  5,  1,  3,  1,  5,  5,  1,  3],
     [ 4,  4,  2,  0,  4,  0,  2,  2,  0,  4,  0,  2],
     [ 5,  5,  3,  1,  3,  5,  1,  3,  1,  3,  5,  1],
     [ 6,  6, 10,  8,  8, 10,  6, 10,  8,  8, 10,  6],
     [ 7,  7, 11,  9,  7,  9, 11, 11,  9,  7,  9, 11],
     [ 8,  8,  6, 10,  6,  8, 10,  6, 10,  6,  8, 10],
     [ 9,  9,  7, 11, 11,  7,  9,  7, 11, 11,  7,  9],
     [10, 10,  8,  6, 10,  6,  8,  8,  6, 10,  6,  8],
     [11, 11,  9,  7,  9, 11,  7,  9,  7,  9, 11,  7]],
    [[ 0,  4,  2],
     [ 1,  5,  3],
     [ 2,  0,  4],
     [ 3,  1,  5],
     [ 4,  2,  0],
     [ 5,  3,  1],
     [ 6, 10,  8],
     [ 7, 11,  9],
     [ 8,  6, 10],
     [ 9,  7, 11],
     [10,  8,  6],
     [11,  9,  7]],
    None
]
subgroup_complex_test_set = [False, False, True, None]
subgroup_order_test_set = [2,12,3,None]

@pytest.mark.parametrize("i", [i for i in range(len(subgroup_fns))])
def test_Symtext_subgroup_symtext(i):
    mol = molsym.Molecule.from_file("test/xyz/"+subgroup_fns[i]+".xyz")
    mol = molsym.symmetrize(mol)
    symtext = molsym.Symtext.from_molecule(mol)
    #assert (mol_test_set[i][0] == mol.atoms).all()
    #assert np.isclose(mol_test_set[i][1], mol.coords).all()
    try:
        symtext = symtext.subgroup_symtext(subgroup_pgs[i])
        assert symtext.pg.str == subgroup_pgs[i]
        assert (symtext.atom_map == subgroup_atom_map_test_set[i]).all()
        assert subgroup_complex_test_set[i] == symtext.complex
        assert subgroup_order_test_set[i] == symtext.order
    except Exception:
        assert i == 3 # This being the test case that should fail
