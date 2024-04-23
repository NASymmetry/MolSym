from .point_group import PointGroup
from molsym.symtools import normalize
from .multiplication_table import *
import re
from .general_irrep_mats import Symel

def generate_symel_to_class_map(symels, ctab):
    pg = PointGroup.from_string(ctab.name)
    if pg.n is not None:
        ns = pg.n>>1 # pg.n floor divided by 2
    ncls = len(ctab.classes)
    nsymel = len(symels)
    class_map = np.zeros(nsymel, dtype=np.int32)
    class_map[0] = 0 # E is always first
    if pg.family == "C":
        if pg.subfamily == "s" or pg.subfamily == "i":
            class_map[1] = 1
        elif pg.subfamily == "h":
            if pg.n % 2 == 0:
                class_map[3:pg.n+2] = [i for i in range(1,pg.n)] # [2:pg.n] # C_n
                class_map[2] = pg.n # i
                class_map[1] = pg.n+ns # σh
                for i in range(pg.n+2,2*pg.n):# = pg.n+3:2*pg.n # S_n
                    if i > 3*ns:
                        class_map[i] = i-ns
                    else:
                        class_map[i] = i+ns-1
            else:
                for i in range(1,pg.n): # = 2:pg.n+1 # C_n
                    class_map[i+1] = i
                class_map[1] = pg.n # σh
                for i in range(pg.n+1,2*pg.n): # = pg.n+2:2*pg.n # S_n
                    class_map[i] = i
        elif pg.subfamily == "v":
            # The last class is σv (and then σd if n is even), and the last symels are also these!
            cn_class_map(class_map, pg.n, 0, 0)
            if pg.n % 2 == 0:
                class_map[-pg.n:-ns] = ncls-2
                class_map[-ns:] = ncls-1
            else:
                class_map[-pg.n:] = ncls-1
        else:
            class_map[1:] = [i for i in range(1,nsymel)] # 2:nsymel
    elif pg.family == "S":
        if pg.n % 4 == 0:
            for i in range(1,pg.n): # = 2:pg.n
                if i <= ns-1:
                    class_map[i] = 2*i
                else:
                    class_map[i] = 2*(i-ns)+1
        else:
            class_map[1] = ns # i
            class_map[2:ns+1] = [i for i in range(1,ns)] # 2:ns # C_n
            for i in range(ns+1,pg.n): # = ns+2:pg.n # S_n
                if i > ns+(pg.n>>2):
                    class_map[i] = i-(pg.n>>2)
                else:
                    class_map[i] = i+(pg.n>>2)
    elif pg.family == "D":
        if pg.subfamily == "h":
            if pg.n % 2 == 0:
                class_map[1] = ncls-3 # σh
                class_map[2] = (ncls>>1) # i
                cn_class_map(class_map, pg.n, 2, 0) # Cn
                class_map[pg.n+2:3*ns+2] = ns+1 # C2'
                class_map[3*ns+2:2*pg.n+2] = ns+2 # C2''
                for i in range(2*pg.n+2,3*pg.n): # = 2*pg.n+3:3*pg.n+1 # Sn
                    if i > 3*pg.n-ns:
                        class_map[i] = i-2*pg.n+3
                    else:
                        class_map[i] = 3*pg.n+4-i
                # The result of C2'×i changes depending if pg.n ≡ 0 (mod 4)
                # but also D2h doesn't need to be flipped because I treated it special
                if pg.n % 4 == 0 or pg.n == 2:
                    class_map[-pg.n:-ns] = ncls-2 # σv
                    class_map[-ns:] = ncls-1 # σd
                else:
                    class_map[-pg.n:-ns] = ncls-1 # σv
                    class_map[-ns:] = ncls-2 # σd
            else:
                class_map[1] = (ncls>>1)
                cn_class_map(class_map, pg.n, 1, 0)
                class_map[pg.n+1:2*pg.n+1] = ns+1
                cn_class_map(class_map, pg.n, 2*pg.n, ns+2)
                class_map[-1-pg.n+1:] = ncls-1
        elif pg.subfamily == "d":
            if pg.n % 2 == 0:
                cn_class_map(class_map, pg.n, 0, 0) # Cn
                class_map[1:pg.n] = 2*class_map[1:pg.n] # 2*class_map[2:pg.n].-1 # Reposition Cn
                cn_class_map(class_map, pg.n+1, pg.n-1, 0) # Sn
                class_map[pg.n:2*pg.n] = 2*class_map[pg.n:2*pg.n]-1 # 2*(class_map[pg.n+1:2*pg.n].-1) # Reposition Sn
                class_map[-2*pg.n:-pg.n] = ncls-2 # C2'
                class_map[-pg.n:] = ncls-1 # σd
            else:
                class_map[1] = ncls>>1 # i
                cn_class_map(class_map, pg.n, 1, 0) # Cn
                for i in range(pg.n+1,2*pg.n): # = pg.n+2:2*pg.n # Sn
                    if i > pg.n+ns:
                        class_map[i] = i+2-pg.n
                    else:
                        class_map[i] = 2*pg.n+2-i
                class_map[-2*pg.n:-pg.n] = ns+1
                class_map[-pg.n:] = ncls-1 # σd
        else:
            cn_class_map(class_map, pg.n, 0, 0) # Cn
            if pg.n % 2 == 0:
                class_map[-pg.n:-ns] = ncls-2 # Cn'
                class_map[-ns:] = ncls-1 # Cn''
            else:
                class_map[-pg.n:] = ncls-1 # Cn
    else:
        if pg.family == "T":
            if pg.subfamily == "h":
                class_map = np.array([0,1,2,1,2,1,2,1,2,3,3,3,4,5,6,5,6,5,6,5,6,7,7,7])
            elif pg.subfamily == "d":
                class_map = np.array([0,1,1,1,1,1,1,1,1,2,2,2,4,4,4,4,4,4,3,3,3,3,3,3])
            else:
                class_map = np.array([0,1,2,1,2,1,2,1,2,3,3,3])
        elif pg.family == "O":
            if pg.subfamily == "h":
                class_map = np.array([0,3,4,3,3,4,3,3,4,3,1,1,1,1,1,1,1,1,2,2,2,2,2,2,5,6,8,6,6,8,6,6,8,6,7,7,7,7,7,7,7,7,9,9,9,9,9,9])
            else:
                class_map = np.array([0,1,2,1,1,2,1,1,2,1,3,3,3,3,3,3,3,3,4,4,4,4,4,4])
        elif pg.family == "I":
            if pg.subfamily == "h":
                class_map = np.array([0,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                             5,6,7,7,6,6,7,7,6,6,7,7,6,6,7,7,6,6,7,7,6,6,7,7,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9])
            else:
                class_map = np.array([0,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4])
        else:
            raise Exception(f"An invalid point group has been given or unexpected parsing of the point group string has occured: {pg.str}")
    return class_map

def cn_class_map(class_map, n, idx_offset, cls_offset):
    for i in range(1,n): # = 2:n
        if i > (n>>1):
            class_map[i+idx_offset] = n-i+cls_offset
        else:
            class_map[i+idx_offset] = i+cls_offset
    return class_map

def rotate_mol_to_symels(mol, paxis, saxis):
    if np.isclose(np.linalg.norm(paxis), 0.0, atol=global_tol): 
        # Symmetry is C1 and paxis not defined, just return mol
        rmat = rmat_inv = np.eye(3)
        return mol, rmat, rmat_inv
    z = paxis
    if np.isclose(np.linalg.norm(saxis), 0.0, atol=global_tol): 
        # Find a trial vector that works
        x = None
        for trial_vec in np.eye(3):
            x = np.cross(trial_vec, z)
            if not np.isclose(np.linalg.norm(x), 0, atol=global_tol):
                x = normalize(x)
                break
        y = normalize(np.cross(z, x))
    else:
        x = saxis
        y = np.cross(z,x)
    rmat = np.column_stack((x,y,z)) # This matrix rotates z to paxis, etc., ...
    rmat_inv = rmat.T # ... so invert it to take paxis to z, etc.
    new_mol = mol.transform(rmat_inv)
    return new_mol, rmat, rmat_inv

def get_atom_mapping(mol, symels):
    # symels after transformation
    amap = np.zeros((mol.natoms, len(symels)), dtype=int)
    for atom in range(mol.natoms):
        for (s, symel) in enumerate(symels):
            w = where_you_go(mol, atom, symel)
            if w is not None:
                amap[atom,s] = w
            else:
                raise Exception(f"Atom {atom} not mapped to another atom under symel {symel}")
    return amap

def get_linear_atom_mapping(mol, pg):
    amap = np.array([atom for atom in range(mol.natoms)], dtype=int).reshape((mol.natoms,1))
    if pg.family == "D":
        ungerade_map = np.zeros((mol.natoms), dtype=int)
        for atom in range(mol.natoms):
            w = where_you_go(mol, atom, Symel("i", None, -1*np.eye(3), None, None, None))
            if w is not None:
                ungerade_map[atom] = w
            else:
                raise Exception(f"Atom {atom} not mapped to another atom under symel i")
        return np.column_stack((amap, ungerade_map))
    return amap

def where_you_go(mol, atom, symel):
    ratom = np.dot(symel.rrep, mol.coords[atom,:].T)
    for i in range(mol.natoms):
        if np.isclose(mol.coords[i,:], ratom, atol=mol.tol).all():
            return i
    return None

def get_class_name(symels_in_class):
    if "^" in symels_in_class[0].symbol:
        rot_order = []
        for symel in symels_in_class:
            s = re.search(r"\^(\d+)", symel.symbol)
            if s:
                rot_order.append(int(s.groups()[0]))
            else:
                rot_order.append(1)
        pickem = symels_in_class[np.argmin(rot_order)].symbol
    else:
        pickem = symels_in_class[0].symbol
    return re.sub(r"\(\w+\)", "", pickem)

def irrep_sort_idx(irrep_str):
    rsult = 0
    # g and ' always go first
    gchk = r"g"
    ppchk = r"''"
    gm = gchk in irrep_str
    pm = ppchk in irrep_str
    if gm:
        rsult += 0
    elif pm:
        rsult += 10000
    else:
        rsult += 1000
    irrep_letter = irrep_str[1] # the letter
    irrep_num_rgx = r"(\d+)"
    mn = re.match(irrep_num_rgx, irrep_str)
    if mn is not None:
        rsult += int(mn.groups()[0])
    if irrep_letter == 'A':
        rsult += 0
    elif irrep_letter == 'B':
        rsult += 100
    elif irrep_letter == 'E':
        rsult += 200
    elif irrep_letter == 'T':
        rsult += 300
    elif irrep_letter == 'G':
        rsult += 400
    elif irrep_letter == 'H':
        rsult += 500
    else:
        raise Exception(f"Invalid irrep order: {irrep_letter}")
    return rsult
