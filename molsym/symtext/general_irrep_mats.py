import numpy as np
from numpy.linalg import matrix_power
from dataclasses import dataclass
from .point_group import PointGroup
from ..symtools import *
from copy import deepcopy
from .symel import generate_T, generate_Th, generate_Td, generate_O, generate_Oh, generate_I, generate_Ih
from .irrep_mats import irrm_T, irrm_Th, irrm_Td, irrm_O, irrm_Oh, irrm_I, irrm_Ih

np.set_printoptions(precision=3, threshold=np.inf, linewidth=14000, suppress=True)
# New Symel definition!
@dataclass
class Symel():
    """
    New Symel definition that helps with the arbitrary generation of irreducible representation matrices.
    """
    symbol:str
    vector:np.array # Not defined for E or i, axis vector for Cn and Sn, plane normal vector for sigma
    rrep:np.array
    m:int
    n:int
    O:str # Options: E, sigma_v, C_2', i, sigma_h
    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True, formatter={"all":lambda x: f"{x:8.5f}"}):
            return f"\nSymbol: {self.symbol:>10s}: [{self.rrep[0,:]},{self.rrep[1,:]},{self.rrep[2,:]}]"
    def __repr__(self) -> str:
        return self.__str__()
    def __eq__(self, other):
        return self.symbol == other.symbol and np.isclose(self.rrep,other.rrep,atol=1e-10).all()

    @classmethod
    def oldSymel(cls, symbol, vector, rrep):
        return cls(symbol, vector, rrep, None, None, None)

@dataclass 
class Irrep():
    """
    Datastructure for irreducible representation matrices.
    """
    symbol:str
    idx:int # If multiple with same name ("A", "B", "E", etc.)
    sub_idx:int # index if separably degenerate
    d:int # dimension

def pg_to_symels(PG):
    """
    Takes in a point group string and returns the associated symmetry elements, irreducible representations, and irreducible representation matrices.

    :type PG: str
    :rtype: (List[molsym.Symel], List[molsym.Irrep], Dict{Irrep.symbol:NumPy array of shape (nsymel, irrep.d, irrep.d)})
    """
    greek = ["Pi", "Delta", "Phi", "Gamma", "Eta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Omicron"]
    pg = PointGroup.from_string(PG)
    argerr = f"An invalid point group has been given or unexpected parsing of the point group string has occured: {pg.str}"
    z = np.array([0,0,1])
    i = Symel("i", None, inversion_matrix(), 0, 0, "i")
    sh = Symel("sigma_h", np.array([0,0,1]), reflection_matrix(z), 0, 0, "sigma_h")
    if pg.is_linear:
        if pg.family == "C":
            symels = [Symel("E", None, np.eye(3), None, None, None),
                      Symel("C", z, np.eye(3), 0, 0, "E"), 
                      Symel("sigma_v", [1,0,0], reflection_matrix([1,0,0]), 0, 0, "sigma_v")]
            irreps = [Irrep("Sigma^+", None, None, 1), Irrep("Sigma^-", None, None, 1)]
            irreps += [Irrep(greek[i], i, None, 2) for i in range(len(greek))]
            irrep_mats = None
            return symels, irreps, irrep_mats
        elif pg.family == "D":
            symels = [Symel("E", None, np.eye(3), None, None, None), 
                      Symel("C", z, np.eye(3), None, None, None), 
                      Symel("sigma_v", [1,0,0], reflection_matrix([1,0,0]), None, None, None),
                      Symel("i", None, -1*np.eye(3), None, None, None),
                      Symel("S", z, reflection_matrix(z), None, None, None),
                      Symel("C_2'", [1,0,0], Cn([1,0,0], 2), None, None, None)]
            irreps = [Irrep("Sigma_g^+", None, None, 1), Irrep("Sigma_g^-", None, None, 1)]
            irreps += [Irrep(greek[i]+"_g", i, None, 2) for i in range(len(greek))]
            irreps += [Irrep("Sigma_u^+", None, None, 1), Irrep("Sigma_u^-", None, None, 1)]
            irreps += [Irrep(greek[i]+"_u", i, None, 2) for i in range(len(greek))]
            irrep_mats = None
            return symels, irreps, irrep_mats
    if pg.n is not None:
        n_is_even = (pg.n % 2 == 0)
        n_is_doubleeven = (pg.n % 4 == 0)
    if pg.family == "C":
        if pg.subfamily == "h":
            cn_symels, cn_irreps, cn_irrep_mats = Zn(pg.n, "C")
            if n_is_even:
                return direct_product(cn_symels, cn_irreps, cn_irrep_mats, i, "g", "u")
            else:
                return direct_product(cn_symels, cn_irreps, cn_irrep_mats, sh, "'", "''")
        elif pg.subfamily == "v":
            return Dihn(pg.n, "C", "sigma_v")
        elif pg.subfamily == "s":
            symels = [Symel("E", None, np.eye(3), 0, None, "E"), sh]
            irreps = [Irrep("A'", 1, None, 1), Irrep("A''", 1, None, 1)]
            irrep_mats =  {"A'": np.array([[[1]],[[1]]]), 
                           "A''":np.array([[[1]],[[-1]]])}
            return symels, irreps, irrep_mats
        elif pg.subfamily == "i":
            symels = [Symel("E", None, np.eye(3), 0, None, "E"), i]
            irreps = [Irrep("Ag", 1, None, 1), Irrep("Au", 1, None, 1)]
            irrep_mats =  {"Ag": np.array([[[1]],[[1]]]), 
                           "Au":np.array([[[1]],[[-1]]])}
            return symels, irreps, irrep_mats
        elif pg.subfamily is None:
            # Cn branch
            return Zn(pg.n, "C")
        else:
            raise Exception(argerr)
    elif pg.family == "D":
        if pg.subfamily == "h":
            dn_symels, dn_irreps, dn_irrep_mats = Dihn(pg.n, "C", "C_2'")
            if n_is_even:
                return direct_product(dn_symels, dn_irreps, dn_irrep_mats, i, "g", "u")
            else:
                return direct_product(dn_symels, dn_irreps, dn_irrep_mats, sh, "'", "''")
        elif pg.subfamily == "d":
            if n_is_even:
                return Dihn(pg.n*2, "S", "C_2'")
            else:
                dn_symels, dn_irreps, dn_irrep_mats = Dihn(pg.n, "C", "C_2'")
                return direct_product(dn_symels, dn_irreps, dn_irrep_mats, i, "g", "u")
        elif pg.subfamily is None:
            return Dihn(pg.n, "C", "C_2'")
        else:
            raise Exception(argerr)
    elif pg.family == "S":
        if pg.subfamily is None and n_is_even:
            if n_is_doubleeven:
                return Zn(pg.n, "S")
            else:
                cn_symels, cn_irreps, cn_irrep_mats = Zn(pg.n>>1, "C")
                return direct_product(cn_symels, cn_irreps, cn_irrep_mats, i, "g", "u")
        else:
            raise Exception(argerr)
    else:
        if pg.family == "T":
            if pg.subfamily == "h":
                symels = generate_Th()
                irreps = [Irrep("A_g",1,None,1), Irrep("E_g(1)",1,1,1), 
                          Irrep("E_g(2)",1,2,1), Irrep("T_g",1,None,3),
                          Irrep("A_u",1,None,1), Irrep("E_u(1)",1,1,1), 
                          Irrep("E_u(2)",1,2,1), Irrep("T_u",1,None,3)]
                irrep_mats = irrm_Th
                return symels, irreps, irrep_mats
            elif pg.subfamily == "d":
                symels = generate_Td()
                irreps = [Irrep("A_1",1,None,1), Irrep("A_2",1,None,1), 
                          Irrep("E",1,None,2), Irrep("T_1",1,None,3),
                          Irrep("T_2",2,None,3)]
                irrep_mats = irrm_Td
                return symels, irreps, irrep_mats
            else:
                symels = generate_T()
                irreps = [Irrep("A",None,None,1), Irrep("E(1)",1,1,1), 
                          Irrep("E(2)",1,2,1), Irrep("T",None,None,3)]
                irrep_mats = irrm_T
                return symels, irreps, irrep_mats
        elif pg.family == "O":
            if pg.subfamily == "h":
                symels = generate_Oh()
                irreps = [Irrep("A_1g",1,None,1), Irrep("A_2g",1,None,1), 
                          Irrep("E_g",1,None,2), Irrep("T_1g",1,None,3),
                          Irrep("T_2g",2,None,3),
                          Irrep("A_1u",1,None,1), Irrep("A_2u",1,None,1), 
                          Irrep("E_u",1,None,2), Irrep("T_1u",1,None,3),
                          Irrep("T_2u",2,None,3)]
                irrep_mats = irrm_Oh
                return symels, irreps, irrep_mats
            else:
                symels = generate_O()
                irreps = [Irrep("A_1",1,None,1), Irrep("A_2",1,None,1), 
                          Irrep("E",1,None,2), Irrep("T_1",1,None,3),
                          Irrep("T_2",2,None,3)]
                irrep_mats = irrm_O
                return symels, irreps, irrep_mats
        elif pg.family == "I":
            if pg.subfamily == "h":
                symels = generate_Ih()
                irreps = [Irrep("A_g",1,None,1), 
                          Irrep("T_1g",1,None,3), Irrep("T_2g",2,None,3),
                          Irrep("G_g",1,None,4), Irrep("H_g",1,None,5),
                          Irrep("A_u",1,None,1), 
                          Irrep("T_1u",1,None,3), Irrep("T_2u",2,None,3),
                          Irrep("G_u",1,None,4), Irrep("H_u",1,None,5)]
                irrep_mats = irrm_Ih
                return symels, irreps, irrep_mats
            else:
                symels = generate_I()
                irreps = [Irrep("A",1,None,1), 
                          Irrep("T_1",1,None,3), Irrep("T_2",2,None,3),
                          Irrep("G",1,None,4), Irrep("H",1,None,5)]
                irrep_mats = irrm_I
                return symels, irreps, irrep_mats
        else:
            raise Exception(argerr)
    return 0

def Zn(n, generator):
    """
    Builds symmetry elements, irreducible representations, and irreducible representation matrices for cyclic point groups.

    :type n: int
    :type generator: str

    :rtype: (List[molsym.Symel], List[molsym.Irrep], Dict{Irrep.symbol:NumPy array of shape (nsymel, irrep.d, irrep.d)})
    """
    # cyclic group generated by generator
    # Irreps: A1, A2, (B1, B2), E_1(1), E_1(2), E_2(1,2), ..., E_floor((n-1)/2)(1,2)
    x,y,z = np.eye(3)
    e = Symel("E", None, np.eye(3, dtype=np.float64), 0, n, "E")
    symels = [e]
    # Generate irrep labels
    irreps = [Irrep("A",None,None,1)]
    if n % 2 == 0 and n > 1:
        irreps.append(Irrep("B",None,None,1))
    i = 2
    while i < n:
        irreps.append(Irrep(f"E_{i>>1}(1)", i>>1, 1, 1))
        irreps.append(Irrep(f"E_{i>>1}(2)", i>>1, 2, 1))
        i += 2
    if n < 5:
        for irrep in irreps:
            irrep.symbol = irrep.symbol.replace("_1","")
    # Generate symels
    if generator == "C" and n>1:
        cn1 = Symel(f"C_{n}", z, Cn(z, n), 1, n, "E")
        symels.append(cn1)
        for m in range(2,n):
            a,b = reduce(n, m)
            if b == 1:
                symb = f"C_{a}"
            else:
                symb = f"C_{a}^{b}"
            cnm = Symel(symb, z, matrix_power(Cn(z,n),m), m, n, "E")
            symels.append(cnm)
    elif generator == "S":
        sn1 = Symel(f"S_{n}", z, Sn(z, n), 1, n, "E")
        symels.append(sn1)
        for m in range(2,n):
            a,b = reduce(n, m)
            if m % 2 == 0:
                symb = f"C_{a}^{b}"
            else:
                symb = f"S_{a}^{b}"
            if b == 1:
                symb = symb.replace("^1", "")
            cnm = Symel(symb, z, matrix_power(Sn(z,n),m), m, n, "E")
            symels.append(cnm)

    irrep_mats = {}
    if n > 2:
        im_dtype = np.complex128
    else:
        im_dtype = np.float64
    for irrep in irreps:
        irrep_mats[irrep.symbol] = np.zeros((n,1,1), dtype=im_dtype)
    for idx, symel in enumerate(symels):
        for irrep in irreps:
            if irrep.symbol == "A":
                r = 1
            elif irrep.symbol == "B":
                r = (-1)**(symel.m)
            else:
                r = np.exp(2*np.pi*irrep.idx*symel.m*1j/n)
                if irrep.sub_idx == 2:
                    r = np.conj(r)
            irrep_mats[irrep.symbol][idx,0,0] = r
    return symels, irreps, irrep_mats

def omega(m, n):
    gcd = np.gcd(m, n)
    l = (m/gcd) + (n/gcd) * (1-((m/gcd)%2))
    return int(l), int(n/gcd)

def mult_iCnm(m, n):
    a = (2*m+n) % (2*n)
    return omega(a, 2*n)

def mult_sigmahCnm(m, n):
    return omega(m, n)

def mult_CSC2sigma(m, n, pre, post):
    # Operator: "C_2", "sigma"
    if pre == "C":
        if post == "C_2":
            even_odd = ["'", "''"]
        elif post == "sigma":
            even_odd = ["_v", "_d"]
    else:
        if post == "C_2":
            post = "sigma"
            even_odd = ["_v", "_d"]
    if n % 2 == 0:
        if pre == "iC":
            if n % 4 == 0:
                label = even_odd[m%2]
                a = ((n>>2) + (m>>1)) % (n>>1)
            else:
                label = even_odd[(m+1)%2]
                a = ((n>>2) + ((m+1)>>1)) % (n>>1)
            return post + label + f"({a})"
        else:
            label = even_odd[m%2]
            a = (m>>1) % (n>>1)
            return post + label + f"({a})"
    else:
        if pre == "iC":
            label = "_d"
            a = ((n>>1) + m) % n
        else:
            label = even_odd[0]
            a = ((((n>>1)+1) * (m % 2))+(m>>1)) % n
        return post + label + f"({a})"

def Dihn(n, generator, symel_tag):
    """
    Builds symmetry elements, irreducible representations, and irreducible representation matrices for dihedral isomorphic point groups.

    :param generator: Symmetry element that generates half of the point group
    :param symel_tag: Operation that generates the other half of the point group
    :type n: int
    :type generator: str
    :type symel_tag: str

    :rtype: (List[molsym.Symel], List[molsym.Irrep], Dict{Irrep.symbol:NumPy array of shape (nsymel, irrep.d, irrep.d)})
    """
    x, y, z = np.eye(3)
    # Generate irrep labels
    irreps = [Irrep("A_1", 1, None, 1), Irrep("A_2", 2, None, 1)]
    if n % 2 == 0 and n > 1:
        irreps.append(Irrep("B_1", 1, None, 1))
        irreps.append(Irrep("B_2", 2, None, 1))
    i = 2
    while i < n:
        irreps.append(Irrep(f"E_{i>>1}", i>>1, None, 2))
        i += 2
    if n < 5:
        for irrep in irreps:
            if irrep.d == 2:
                irrep.symbol = irrep.symbol.replace("_1","")
    # Generate symels
    e = Symel("E", None, np.eye(3, dtype=np.float64), 0, n, "E")
    symels = [e]
    if generator == "C":
        if symel_tag == "sigma_v":
            # First reflection contains the x-axis so the normal vector is y-axis
            # Rotated reflections trail the associated rotation by theta/2
            post = "sigma"
            on0 = Symel("sigma_v(0)", y, reflection_matrix(y), 0, n, "sigma_v")
            tag_re = ("v", "d")
        elif symel_tag == "C_2'":
            # First C_2' is about the x-axis
            post = "C_2"
            on0 = Symel(f"C_2'(0)", x, Cn(x, 2), 0, n, "C_2'")
            tag_re = ("'", "''")
        if n % 2 == 0:
            rsymb = symel_tag.replace(tag_re[0], tag_re[1]) + "(0)"
        else:# symel_tag == "sigma_v":
            rsymb = symel_tag+f"({(n>>1)+1})"
        symels.append(on0)
        for m in range(1,n):
            a,b = reduce(n, m)
            if b == 1:
                symb = f"C_{a}"
            else:
                symb = f"C_{a}^{b}"
            rot = matrix_power(Cn(z, n), m)
            cnm = Symel(symb, z, matrix_power(Cn(z, n), m), m, n, "E")
            symels.insert(m, cnm)
            rsymb = mult_CSC2sigma(m, n, "C", post)
            new_vector = np.dot(matrix_power(Cn(z, 2*n), m), on0.vector)
            onm = Symel(rsymb, new_vector, rot @ on0.rrep, m, n, symel_tag)
            symels.append(onm)
    elif generator == "S":
        sn1 = Symel(f"S_{n}", z, Sn(z, n), 1, n, "E")
        symels.append(sn1)
        on0 = Symel("C_2'(0)", x, Cn(x,2), 0, n, "C_2'")
        symels.append(on0)
        v = np.dot(Cn(z,2*n), y)
        on1 = Symel("sigma_d(0)", v, reflection_matrix(v), 1, n, "C_2'")
        symels.append(on1)
        for m in range(2,n):
            a,b = reduce(n, m)
            if m % 2 == 0:
                symb = f"C_{a}^{b}"
                rsymb = f"C_2'({m>>1})"
                rot = matrix_power(Cn(z, n), m)
                vec = np.dot(matrix_power(Cn(z, n), m>>1), on0.vector)
                rrep = rot @ on0.rrep
                insert_idx = -(m>>1)
            else:
                symb = f"S_{a}^{b}"
                rsymb = f"sigma_d({m>>1})"
                rot = matrix_power(Cn(z, n), m-1)
                
                vec = np.dot(matrix_power(Cn(z, n), m>>1), on1.vector)
                #rrep = reflection_matrix(vec)
                rrep = rot @ on1.rrep
                insert_idx = len(symels)+1
            if b == 1:
                symb = symb.replace("^1", "")
            snm = Symel(symb, z, matrix_power(Sn(z, n), m), m, n, "E")
            symels.insert(m, snm)
            onm = Symel(rsymb, vec, rrep, m, n, symel_tag)
            symels.insert(insert_idx, onm)
    else:
        raise Exception(f"Invalid generator {generator}")
    irrep_mats = {}
    gen_mat = np.array([[1,0],[0,-1]])
    for irrep in irreps:
        irrep_mats[irrep.symbol] = np.zeros((2*n, irrep.d, irrep.d), dtype=np.float64)
    for idx, symel in enumerate(symels):
        for irrep in irreps:
            if irrep.symbol == "A_1":
                r = 1
            elif irrep.symbol == "A_2":
                r = -1 if symel_tag in symel.O else 1
            elif irrep.symbol == "B_1":
                r = 1 if symel.m % 2 == 0 else -1
            elif irrep.symbol == "B_2":
                ra = -1 if symel_tag in symel.O else 1
                rb = 1 if symel.m % 2 == 0 else -1
                r = ra * rb
            else:
                rc = np.cos(2*np.pi*irrep.idx*symel.m/n)
                rs = np.sin(2*np.pi*irrep.idx*symel.m/n)
                r = np.array([[rc, -rs],[rs, rc]])
                if symel_tag in symel.O:
                    r = r @ gen_mat
            irrep_mats[irrep.symbol][idx,:,:] = r
    if n == 2 and symel_tag == "C_2'":
        irrep_mats["B_3"] = irrep_mats.pop("B_2")
        irrep_mats["B_2"] = irrep_mats.pop("B_1")
        irrep_mats["B_1"] = irrep_mats.pop("A_2")
        irreps[1].symbol = "B_1"
        irreps[2].symbol = "B_2"
        irreps[3].symbol = "B_3"
    return symels, irreps, irrep_mats

def direct_product(symels, irreps, irrep_mats, dpsymel, irrep_tag1, irrep_tag2):
    """
    Builds symmetry elements, irreducible representations, and irreducible representation matrices from a direct product of a base point group with a point group of order 2.

    :param symels: Symmetry elements of base point group
    :param irreps: Irreducible representations of base point group
    :param irrep_mats: Irreducible representation matrices of base point group
    :param dpsymel: Non-identity symmetry element of point group of order 2
    :param irrep_tag1: Tag to be added to irreducible representation symbols (e.g. "g" or "'")
    :param irrep_tag2: Tag to be added to irreducible representation symbols (e.g. "u" or "''")
    :type symels: List[molsym.Symel]
    :type irreps: List[molsym.Irrep]
    :type irrep_mats: Dict{Irrep.symbol:NumPy array of shape (nsymel, irrep.d, irrep.d)}
    :type dpsymel: molsym.Symel
    :type irrep_tag1: str
    :type irrep_tag2: str
    :rtype: (List[molsym.Symel], List[molsym.Irrep], Dict{Irrep.symbol:NumPy array of shape (nsymel, irrep.d, irrep.d)})
    """
    x, y, z = np.eye(3)
    newsymels = [symel for symel in symels]
    for symel in symels:
        new_rrep = symel.rrep @ dpsymel.rrep
        if dpsymel.symbol == "i":
            if symel.O == "E":
                # Z proper rotations
                b, a = mult_iCnm(symel.m, symel.n)
                if a == 2:
                    new_symbol = "i"
                    new_vector = None
                elif a == 1:
                    new_symbol = "sigma_h"
                    new_vector = z
                else:
                    if b == 1:
                        new_symbol = f"S_{a}"
                    else:
                        new_symbol = f"S_{a}^{b}"
                    new_vector = symel.vector
                new_O = dpsymel.symbol
            elif symel.O == "C_2'":
                #print(symel)
                new_symbol = mult_CSC2sigma(symel.m, symel.n, "iC", "C_2")
                #if symel.n % 2 == 0:
                #    if symel.n % 4 == 0:
                #        sigma_tags = ("v", "d")
                #    else:
                #        sigma_tags = ("d", "v")
                #    if "''" in symel.symbol:
                #        new_symbol = f"sigma_{sigma_tags[1]}({(((symel.n>>1)+symel.m)%symel.n)>>1})"
                #    else:
                #        new_symbol = f"sigma_{sigma_tags[0]}({(((symel.n>>1)+symel.m)%symel.n)>>1})"
                #else:
                #    new_symbol = f"sigma_d({((symel.n>>1)+symel.m)%symel.n})"
                new_vector = symel.vector
                #new_vector = np.dot(matrix_power(Cn(z, 2*symel.n), symel.m), y.T)
                new_O = dpsymel.symbol
        elif dpsymel.symbol == "sigma_h":
            # n is odd
            if symel.O == "E":
                # Z proper rotations
                new_O = "sigma_h"
                b, a = mult_sigmahCnm(symel.m, symel.n)
                if a == 1:
                    new_symbol = "sigma_h"
                    new_vector = z
                else:
                    if b == 1:
                        new_symbol = f"S_{a}"
                    else:
                        new_symbol = f"S_{a}^{b}"
                    new_vector = symel.vector
            elif symel.O == "C_2'":
                # Dnh, n is odd
                new_vector = normalize(np.cross(z, symel.vector))
                new_symbol = symel.symbol.replace("C_2'", "sigma_v")
                new_O = "sigma_v"
        else:
            raise Exception(f"Invalid direct product element: {dpsymel.symbol}")
        ns = Symel(new_symbol, new_vector, new_rrep, symel.m, symel.n, new_O)
        newsymels.append(ns)
        newirreps = deepcopy(irreps)
        new_irrep_mats = {}
        for idx, irrep in enumerate(irreps):
            newirreps[idx].symbol = irrep.symbol+irrep_tag1
            new_irrep_mats[irrep.symbol+irrep_tag1] = np.concatenate((irrep_mats[irrep.symbol], irrep_mats[irrep.symbol]))
            newirrep = Irrep(irrep.symbol+irrep_tag2, irrep.idx, irrep.sub_idx, irrep.d)
            newirreps.append(newirrep)
            new_irrep_mats[newirrep.symbol] = np.concatenate((irrep_mats[irrep.symbol], -1 * irrep_mats[irrep.symbol]))
        
    return newsymels, newirreps, new_irrep_mats
