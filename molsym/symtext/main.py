from .symel_generators import *
from .character_table_generators import *
from molsym.symtools import normalize
from .multiplication_table import *
import re

def pg_to_symels(PG):
    pg = PointGroup.from_string(PG)
    argerr = f"An invalid point group has been given or unexpected parsing of the point group string has occured: {pg.str}"
    symels = [Symel("E", None, np.asarray([[1,0,0],[0,1,0],[0,0,1]]))]
    z_axis = np.array([0,0,1])
    sigma_h = np.asarray([[1,0,0],[0,1,0],[0,0,-1]])
    if pg.family == "C":
        if pg.subfamily == "h":
            symels.append(Symel("sigma_h", z_axis, sigma_h))
            if pg.n % 2 == 0:
                symels.append(Symel("i", None, inversion_matrix()))
            cns = generate_Cn(pg.n)
            sns = generate_Sn(pg.n)
            symels = symels + cns + sns
        elif pg.subfamily == "v":
            cns = generate_Cn(pg.n)
            if pg.n % 2 == 0:
                n = pg.n >> 1
                sigma_ds = generate_sigma_d(n)
            else:
                n = pg.n
                sigma_ds = []
            sigma_vs = generate_sigma_v(pg.n)
            symels = symels + cns + sigma_vs + sigma_ds
        elif pg.subfamily == "s":
            symels.append(Symel("sigma_h", z_axis, sigma_h))
        elif pg.subfamily == "i":
            symels.append(Symel("i", None, inversion_matrix()))
        elif pg.subfamily is None:
            cns = generate_Cn(pg.n)
            symels = symels + cns
        else:
            raise Exception(argerr)
    elif pg.family == "D":
        if pg.subfamily == "h":
            symels.append(Symel("sigma_h", z_axis, sigma_h))
            if pg.n % 2 == 0:
                symels.append(Symel("i", None, inversion_matrix()))
                n = pg.n >> 1
                sigma_ds = generate_sigma_d(n)
                c2ps = generate_C2p(pg.n)
                c2pps = generate_C2pp(pg.n)
                c2s = c2ps + c2pps
            else:
                n = pg.n
                sigma_ds = []
                c2s = generate_C2p(pg.n)
            cns = generate_Cn(pg.n)
            sns = generate_Sn(pg.n)
            sigma_vs = generate_sigma_v(pg.n)
            #c2s = generate_C2(pg.n)
            symels = symels + cns + c2s + sns + sigma_vs + sigma_ds
        elif pg.subfamily == "d":
            if pg.n % 2 == 0:
                c2ps = generate_C2p(pg.n)
                c2pps = generate_C2pp(pg.n)
                c2s = c2ps + c2pps
            else:
                c2s = generate_C2p(pg.n)
                symels.append(Symel("i", None, inversion_matrix()))
            cns = generate_Cn(pg.n)
            sns = generate_Sn(pg.n * 2, S2n=True)
            sigma_ds = generate_sigma_d(pg.n)
            symels = symels + cns + sns + c2s + sigma_ds
        elif pg.subfamily is None:
            cns = generate_Cn(pg.n)
            if pg.n % 2 == 0:
                c2ps = generate_C2p(pg.n)
                c2pps = generate_C2pp(pg.n)
                c2s = c2ps + c2pps
            else:
                c2s = generate_C2p(pg.n)
            symels = symels + cns + c2s
        else:
            raise Exception(argerr)
    elif pg.family == "S":
        if pg.subfamily is None and (pg.n % 2 == 0):
            n = pg.n >> 1
            if n % 2 != 0:
                symels.append(Symel("i", None, inversion_matrix()))
            cns = generate_Cn(n)
            sns = generate_Sn(pg.n, S2n=True)
            symels = symels + cns + sns
        else:
            raise Exception(argerr)
    else:
        if pg.family == "T":
            if pg.subfamily == "h":
                Ths = generate_Th()
                symels = symels + Ths
            elif pg.subfamily == "d":
                Tds = generate_Td()
                symels = symels + Tds
            else:
                Ts = generate_T()
                symels = symels + Ts
        elif pg.family == "O":
            if pg.subfamily == "h":
                Ohs = generate_Oh()
                symels = symels + Ohs
            else:
                Os = generate_O()
                symels = symels + Os
        elif pg.family == "I":
            if pg.subfamily == "h":
                Ihs = generate_Ih()
                symels = symels + Ihs
            else:
                Is = generate_I()
                symels = symels + Is
        else:
            raise Exception(argerr)
    return symels

def pg_to_chartab(PG):
    pg = PointGroup.from_string(PG)
    irreps = []
    if pg.family == "C":
        if pg.subfamily == "s":
            irreps = ["A'","A''"]
            classes = ["E", "sigma_h"]
            chars = np.array([[1.0, 1.0], [1.0, -1.0]])
        elif pg.subfamily == "i":
            irreps = ["Ag","Au"]
            classes = ["E", "i"]
            chars = np.array([[1.0, 1.0], [1.0, -1.0]])
        elif pg.subfamily == "v":
            irreps, classes, chars = Cnv_irr(pg.n)
        elif pg.subfamily == "h":
            irreps, classes, chars = Cnh_irr(pg.n)
        else:
            irreps, classes, chars = Cn_irrmat(pg.n)
    elif pg.family == "D":
        if pg.subfamily == "d":
            irreps, classes, chars = Dnd_irr(pg.n)
        elif pg.subfamily == "h":
            irreps, classes, chars = Dnh_irr(pg.n)
        else:
            irreps, classes, chars = Dn_irr(pg.n)
    elif pg.family == "S":
        irreps, classes, chars = Sn_irr(pg.n)
    else:
        cp3 = np.cos(np.pi/3)
        pr5 = 0.5*(1.0+np.sqrt(5.0))
        mr5 = 0.5*(1.0-np.sqrt(5.0))
        if pg.family == "T":
            if pg.subfamily == "h":
                irreps, classes, chars = (["Ag","Au","Eg","Eu","Tg","Tu"],
                 ["E", "4C_3", "4C_3^2", "3C_2", "i", "S_6", "S_6^5", "3sigma_h"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                  [1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
                  [2.0,  cp3,  cp3,  2.0,  2.0,  cp3,  cp3,  1.0],
                  [2.0,  cp3,  cp3,  2.0, -2.0, -cp3, -cp3, -1.0],
                  [3.0,  0.0,  0.0, -1.0,  1.0,  0.0,  0.0, -1.0],
                  [3.0,  0.0,  0.0, -1.0, -1.0,  0.0,  0.0,  1.0]]))
            elif pg.subfamily == "d":
                irreps, classes, chars = (["A1","A2","E","T1","T2"],
                 ["E", "8C_3", "3C_2", "6S_4", "6sigma_d"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0,  1.0],
                  [1.0,  1.0,  1.0, -1.0, -1.0],
                  [2.0, -1.0,  2.0,  0.0,  0.0],
                  [3.0,  1.0, -1.0,  1.0, -1.0],
                  [3.0, -1.0, -1.0, -1.0,  1.0]]))
            else:
                irreps, classes, chars = (["A","E","T"],
                 ["E", "4C_3", "4C_3^2", "3C_2"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0],
                  [2.0,  cp3,  cp3,  2.0],
                  [3.0,  0.0,  0.0, -1.0]]))
        elif pg.family == "O":
            if pg.subfamily == "h":
                irreps, classes, chars = (["A1g","A2g","Eg","T1g","T2g","A1u","A2u","Eu","T1u","T2u"],
                 ["E", "8C_3", "6C_2", "6C_4", "3C_2", "i", "6S_4", "8S_6", "3sigma_h", "6sigma_d"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                  [1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
                  [2.0, -1.0,  0.0,  0.0,  2.0,  2.0,  0.0, -1.0,  2.0,  0.0],
                  [3.0,  0.0, -1.0,  1.0, -1.0,  3.0,  1.0,  0.0, -1.0, -1.0],
                  [3.0,  0.0,  1.0, -1.0, -1.0,  3.0, -1.0,  0.0, -1.0,  1.0],
                  [1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                  [1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0],
                  [2.0, -1.0,  0.0,  0.0,  2.0, -2.0,  0.0,  1.0, -2.0,  0.0],
                  [3.0,  0.0, -1.0,  1.0, -1.0, -3.0, -1.0,  0.0,  1.0,  1.0],
                  [3.0,  0.0,  1.0, -1.0, -1.0, -3.0,  1.0,  0.0,  1.0, -1.0]]))
            else:
                irreps, classes, chars = (["A1","A2","E","T1","T2"],
                 ["E", "6C_4", "3C_2", "8C_3", "6C_2"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0,  1.0],
                  [1.0, -1.0,  1.0,  1.0, -1.0],
                  [2.0,  0.0,  2.0, -1.0,  0.0],
                  [3.0,  1.0, -1.0,  0.0, -1.0],
                  [3.0, -1.0, -1.0,  0.0,  1.0]]))
        elif pg.family == "I":
            if pg.subfamily == "h":
                irreps, classes, chars = (["Ag","T1g","T2g","Gg","Hg","Au","T1u","T2u","Gu","Hu"],
                 ["E", "12C_5", "12C_5^2", "20C_3", "15C_2", "i", "12S_10", "12S_10^3", "20S_6", "15sigma_"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                  [3.0,  pr5,  mr5,  0.0, -1.0,  3.0,  mr5,  pr5,  0.0, -1.0],
                  [3.0,  mr5,  pr5,  0.0, -1.0,  5.0,  0.0,  0.0, -1.0,  1.0],
                  [4.0, -1.0, -1.0,  1.0,  0.0,  4.0, -1.0, -1.0,  1.0,  0.0],
                  [5.0,  0.0,  0.0, -1.0,  1.0,  5.0,  0.0,  0.0, -1.0,  1.0],
                  [1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                  [3.0,  pr5,  mr5,  0.0, -1.0, -3.0, -mr5, -pr5,  0.0,  1.0],
                  [3.0,  mr5,  pr5,  0.0, -1.0, -3.0, -pr5, -mr5,  0.0,  1.0],
                  [4.0, -1.0, -1.0,  1.0,  0.0, -4.0,  1.0,  1.0, -1.0,  0.0],
                  [5.0,  0.0,  0.0, -1.0,  1.0, -5.0,  0.0,  0.0,  1.0, -1.0]]))
            else:
                irreps, classes, chars = (["A","T1","T2","G","H"],
                 ["E", "12C_5", "12C_5^2", "20C_3", "15C_2"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0,  1.0],
                  [3.0,  pr5,  mr5,  0.0, -1.0],
                  [3.0,  mr5,  pr5,  0.0, -1.0],
                  [4.0, -1.0, -1.0,  1.0,  0.0],
                  [5.0,  0.0,  0.0, -1.0,  1.0]]))
        else:
            raise Exception(f"An invalid point group has been given or unexpected parsing of the point group string has occured: {pg.str}")
    class_orders = grab_class_orders(classes)
    irr_dims = {}
    for (irr_idx,irrep) in enumerate(irreps):
        if pg.n == 1:
            irr_dims[irrep] = int(chars[0])
        else:
            irr_dims[irrep] = int(chars[irr_idx, 0])
    return CharTable(PG, irreps, classes, class_orders, chars, irr_dims)

def grab_class_orders(classes):
    ncls = len(classes)
    class_orders = np.zeros(ncls)
    for i in range(ncls): # = 1:ncls
        class_orders[i] = grab_order(classes[i])
    return class_orders

def grab_order(class_str):
    regex = r"^(\d+)"
    m = re.match(regex, class_str)
    if m is not None:
        return int(m.groups()[0])
    else:
        return 1

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

def rotate_symels_to_mol(symels, paxis, saxis):
    phi, theta, chi = get_euler_angles(paxis, saxis)
    dc = dc_mat(phi, theta, chi)
    dct = deepcopy(dc)
    dct = dct.transpose()
    new_symels = []
    for s in symels:
        new_symels.append(Symel(s.symbol, dct*s.rrep*dc))
    return new_symels

def get_euler_angles(paxis, saxis):
    x = np.array([1.0,0.0,0.0])
    y = np.array([0.0,1.0,0.0])
    z = np.array([0.0,0.0,1.0])
    ynew = np.cross(paxis,saxis)
    zxp = normalize(np.cross(z,paxis))
    if zxp is None:
        phi = 0.0
    else:
        xproj = np.dot(zxp,x)
        if xproj <= 0:
            phi = np.arccos(np.dot(y,zxp))
        else:
            phi = 2*np.pi-np.arccos(np.dot(y,zxp))
    r_phi = rotation_matrix(z,phi)
    yN = np.dot(r_phi,y)
    xN = np.dot(r_phi,x)
    theta = np.arccos(np.dot(z,paxis))
    r_theta = rotation_matrix(yN,theta)
    x2N = np.dot(r_theta,xN)
    Z = np.dot(r_theta,z)
    chi = np.arccos(np.dot(x2N,saxis))
    r_chi = rotation_matrix(Z,chi)
    X = np.dot(r_chi,x2N)
    Y = np.dot(r_chi,yN)
    return phi, theta, chi

def dc_mat(phi, theta, chi):
    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    sc = np.sin(chi)
    cc = np.cos(chi)
    direction_cosine = np.array([[cp*ct*cc-sp*sc, sp*ct*cc+cp*sc, -cc*st], [-cp*ct*sc-sp*cc, -sp*ct*sc+cp*cc, sc*st], [st*cp, st*sp, ct]])
    return direction_cosine

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

def where_you_go(mol, atom, symel):
    ratom = np.dot(symel.rrep, mol.coords[atom,:].T)
    for i in range(mol.natoms):
        if np.isclose(mol.coords[i,:], ratom, atol=mol.tol).all():
            return i
    return None

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
