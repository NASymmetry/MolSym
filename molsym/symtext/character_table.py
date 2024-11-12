import numpy as np
from .point_group import PointGroup
from ..symtools import *
import re

"""
These functions are deprecated

:deprecated:
"""

class CharacterTable():
    def __init__(self, pg, irreps, classes, class_orders, chars, irrep_dims) -> None:
        print("Deprecation Warning: Using deprecated CharacterTable class")
        self.name = pg
        self.irreps = irreps
        self.classes = classes
        self.class_orders = class_orders
        self.characters = chars
        self.irrep_dims = irrep_dims
    def __repr__(self) -> str:
        return f"Character Table for {self.name}\nIrreps: {self.irreps}\nClasses: {self.classes}\nCharacters:\n{self.characters}\n"
    def __eq__(self, other):
        if len(self.irreps) == len(other.irreps) and len(self.classes) == len(other.classes) and np.shape(self.characters)==np.shape(other.characters):
            return (self.irreps == other.irreps).all() and (self.classes == other.classes).all() and np.isclose(self.characters,other.characters,atol=1e-10).all()
        else:
            return False

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
            #irreps, classes, chars = Cn_irrmat(pg.n)
            irreps, classes, chars = Cn_irr_complex(pg.n)
    elif pg.family == "D":
        if pg.subfamily == "d":
            irreps, classes, chars = Dnd_irr(pg.n)
        elif pg.subfamily == "h":
            irreps, classes, chars = Dnh_irr(pg.n)
        else:
            irreps, classes, chars = Dn_irr(pg.n)
    elif pg.family == "S":
        irreps, classes, chars = Sn_irr_complex(pg.n)
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
                  [3.0,  0.0, -1.0,  1.0, -1.0],
                  [3.0,  0.0, -1.0, -1.0,  1.0]]))
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
                 ["E", "12C_5", "12C_5^2", "20C_3", "15C_2", "i", "12S_10", "12S_10^3", "20S_6", "15sigma"],
                 np.array(
                 [[1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                  [3.0,  pr5,  mr5,  0.0, -1.0,  3.0,  mr5,  pr5,  0.0, -1.0],
                  [3.0,  mr5,  pr5,  0.0, -1.0,  3.0,  pr5,  mr5,  0.0, -1.0],
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
            irr_dims[irrep] = int(np.real(chars[irr_idx, 0]))
    return CharacterTable(PG, irreps, classes, class_orders, chars, irr_dims)

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

def c_class_string(classes, pre, r, s):
    "Pushes class string to 'classes' for rotations, but ignores superscript if s is one"
    if s == 1:
        classes.append(pre+f"_{r}")
    else:
        classes.append(pre+f"_{r}^{s}")

def Cn_irrmat(n):
    names = ["A"]
    classes = ["E"]
    for c in range(1,n):
        r,s = reduce(n,c)
        c_class_string(classes, "C", r, s)
    chars = np.ones(n)
    if n % 2 == 0:
        names.append("B")
        bi = np.ones(n)
        #for i=1:n:
        for i in range(n):
            if (i+1) % 2 == 0:
                bi[i] *= -1
        chars = np.vstack((chars, bi))
    if 2 < n < 5:
        # No label associated with E if n < 5
        names.append("E")
        theta = 2*np.pi / n
        v = np.zeros(n)
        for j in range(n):
            v[j] += 2*np.cos(j*theta)
        chars = np.vstack((chars, v))
    elif n >= 5:
        theta = 2*np.pi / n
        l = round(((n-len(names))/2))
        for i in range(l):#= 1:l:
            names.append(f"E{i+1}")
            v = np.zeros(n)
            for j in range(n):
                v[j] += 2*np.cos((i+1)*j*theta)
            chars = np.vstack((chars, v))
    return names, classes, chars

def Cn_irr_complex(n):
    names = ["A"]
    classes = ["E"]
    for c in range(1,n):
        r,s = reduce(n,c)
        c_class_string(classes, "C", r, s)
    chars = np.ones(n)
    if n % 2 == 0:
        names.append("B")
        bi = np.ones(n)
        #for i=1:n:
        for i in range(n):
            if (i+1) % 2 == 0:
                bi[i] *= -1
        chars = np.vstack((chars, bi))
    if 2 < n < 5:
        # No label associated with E if n < 5
        names.append("E_1")
        names.append("E_2")
        theta = np.exp(2*np.pi*1j / n)
        v1 = np.zeros(n, dtype=np.complex128)
        v2 = np.zeros(n, dtype=np.complex128)
        for a in range(n):
            v1[a] += theta**a
            v2[a] += np.conj(theta**a)
        chars = np.vstack((chars, v1))
        chars = np.vstack((chars, v2))
    elif n >= 5:
        theta = 2*np.pi / n
        l = round(((n-len(names))/2))
        for a in range(l):#= 1:l:
            names.append(f"E{a+1}_1")
            names.append(f"E{a+1}_2")
            theta = np.exp(2*np.pi*1j*(a+1) / n)
            v1 = np.zeros(n, dtype=np.complex128)
            v2 = np.zeros(n, dtype=np.complex128)
            for b in range(n):
                v1[b] += theta**b
                v2[b] += np.conj(theta**b)
            chars = np.vstack((chars, v1))
            chars = np.vstack((chars, v2))
    return names, classes, chars

def Cnv_irr(n):
    names, classes, chars = Cn_irrmat(n)
    classes = ["E"]
    if n % 2 == 0:
        for c in range(1,n>>1): # = 1:(n>>1)-1:
            r,s = reduce(n,c)
            c_class_string(classes, "2C", r, s)
        classes.append("C_2")
        if n>>1 == 1:
            classes.append("sigma_v(xz)")
            classes.append("sigma_d(yz)")
        else:
            classes.append(f"{n>>1}sigma_v")
            classes.append(f"{n>>1}sigma_d")
    else:
        for c in range(1,(n>>1)+1): # = 1:n>>1:
            r,s = reduce(n,c)
            c_class_string(classes, "2C", r, s)
        classes.append(f"{n}sigma_v")
    names[0] = "A1"
    names.insert(1, "A2")
    chars = np.vstack((chars[0,:], chars[0,:], chars[1:,:]))
    for i in range(1,n):# = 2:n:
        if i >= n-i:
            break
        #deleteat!(classes, n-i+2)
        chars = chars[:,[j for j in range(np.shape(chars)[1]) if j != n-i]] #chars[:,1:-1.!=n-i+2]
    if n % 2 == 0:
        nirr = round((n/2)+3)
        names[2] = "B1"
        names.insert(3, "B2")
        chars = np.vstack((chars[0:3,:], chars[2,:], chars[3:,:]))
        sigma_v = np.zeros(nirr)
        sigma_d = np.zeros(nirr)
        sigma_v[0:4] = np.array([1,-1,1,-1])
        sigma_d[0:4] = np.array([1,-1,-1,1])
        chars = np.hstack((chars, sigma_v[:,None], sigma_d[:,None]))
    else:
        nirr = round((n-1)/2+2)
        sigma_v = np.zeros(nirr)
        sigma_v[0:2] = np.array([1, -1])
        chars = np.hstack((chars, sigma_v[:,None]))
    return names, classes, chars

def Cnh_irr(n):
    #names, classes, cnchars = Cn_irrmat(n)
    names, classes, cnchars = Cn_irr_complex(n)
    if n % 2 == 0: 
        classes.append("i")
        for i in range(1,n): # = 1:n-1:
            if i == n>>1:
                classes.append("sigma_h")
            else:
                r,s = reduce(n, (i+(n>>1))%n)
                if s % 2 == 0:
                    s += r
                c_class_string(classes, "S", r, s)
    else:
        classes.append("sigma_h")
        for i in range(1,n): # = 1:n-1:
            r,s = reduce(n,i)
            if i % 2 == 0:
                c_class_string(classes, "S", r, s+n)
            else:
                c_class_string(classes, "S", r, s)
    if n % 2 == 0:
        newnames = []
        for i in range(len(names)):# = 1:length(names):
            newnames.append(names[i]+"u")
            names[i] = names[i]+"g"
        names += newnames
        cncharsi = -1 * cnchars
        top = np.hstack((cnchars, cnchars))
        bot = np.hstack((cnchars, cncharsi))
        chars = np.vstack((top, bot))
    else:
        newnames = []
        for i in range(len(names)):# = 1:length(names):
            newnames.append(names[i]+"''")
            names[i] = names[i]+"'"
        names += newnames
        cncharsi = -1 * cnchars
        top = np.hstack((cnchars, cnchars))
        bot = np.hstack((cnchars, cncharsi))
        chars = np.vstack((top, bot))
    return names, classes, chars

def Sn_irr(n):
    if n % 4 == 0:
        names, classes, chars = Cn_irrmat(n)
        for i in range(n): # = 1:n:
            if (i+1) % 2 == 0:
                classes[i] = "S"+classes[i][1:]
    elif n % 2 == 0:
        ni = round(n/2)
        names, classes, cnchars = Cn_irrmat(ni)
        classes = ["E"]
        for i in range(1,n>>1): # = 1:n>>1-1:
            r,s = reduce(n>>1, i)
            c_class_string(classes, "C", r, s)
        classes.append("i")
        for i in range(1,n>>1): # = 1:n>>1-1:
            r,s = reduce(n, ((i<<1)+(n>>1))%n)
            c_class_string(classes, "S", r, s)
        newnames = []
        for i in range(len(names)): # = 1:length(names):
            newnames.append(names[i]+"u")
            names[i] = names[i]+"g"
        names += newnames
        cncharsi = -1 * cnchars
        top = np.hstack((cnchars, cnchars))
        bot = np.hstack((cnchars, cncharsi))
        chars = np.vstack((top, bot))
    else:
        raise Exception("Odd number n for S group")
    return names, classes, chars

def Sn_irr_complex(n):
    if n % 4 == 0:
        names, classes, chars = Cn_irr_complex(n)
        for i in range(n): # = 1:n:
            if (i+1) % 2 == 0:
                classes[i] = "S"+classes[i][1:]
    elif n % 2 == 0:
        ni = round(n/2)
        names, classes, cnchars = Cn_irr_complex(ni)
        classes = ["E"]
        for i in range(1,n>>1): # = 1:n>>1-1:
            r,s = reduce(n>>1, i)
            c_class_string(classes, "C", r, s)
        classes.append("i")
        for i in range(1,n>>1): # = 1:n>>1-1:
            r,s = reduce(n, ((i<<1)+(n>>1))%n)
            c_class_string(classes, "S", r, s)
        newnames = []
        for i in range(len(names)): # = 1:length(names):
            newnames.append(names[i]+"u")
            names[i] = names[i]+"g"
        names += newnames
        cncharsi = -1 * cnchars
        top = np.hstack((cnchars, cnchars))
        bot = np.hstack((cnchars, cncharsi))
        chars = np.vstack((top, bot))
    else:
        raise Exception("Odd number n for S group")
    return names, classes, chars

def Dn_irr(n):
    if n == 2:
        names = ["A", "B1", "B2", "B3"]
        classes = ["E", "C_2(z)", "C_2(y)", "C_2(x)"]
        chars = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
        return names, classes, chars
    names, garbage_classes, chars = Cn_irrmat(n)
    garbage_names, classes, garbage_chars = Cnv_irr(n)
    if n % 2 == 0:
        classes[-2] = classes[-2][0]+"C_2'"
        classes[-1] = classes[-1][0]+"C_2''"
    else:
        classes[-1] = classes[-1][0]+"C_2"
    names[0] = "A1"
    names.insert(1, "A2")
    chars = np.vstack((chars[0,:], chars[0,:], chars[1:,:]))
    for i in range(1,n): # = 2:n:
        if i == n-i or i > n-i:
            break
        #deleteat!(classes, n-i+2)
        chars = chars[:,[j for j in range(np.shape(chars)[1]) if j != n-i]] # chars[:,0:-1.!=n-i+2]
    if n % 2 == 0:
        nirr = round((n/2)+3)
        names[2] = "B1"
        names.insert(3, "B2")
        chars = np.vstack((chars[0:3,:], chars[2,:], chars[3:,:]))
        C2p = np.zeros(nirr)
        C2pp = np.zeros(nirr)
        C2p[0:4] = np.array([1,-1,1,-1])
        C2pp[0:4] = np.array([1,-1,-1,1])
        chars = np.hstack((chars, C2p[:,None], C2pp[:,None]))
    else:
        nirr = round((n-1)/2+2)
        C2p = np.zeros(nirr)
        C2p[0:2] = np.array([1, -1])
        chars = np.hstack((chars, C2p[:,None]))
    return names, classes, chars

def Dnh_irr(n):
    names, classes, dnchars = Dn_irr(n)
    if n % 2 == 0:
        classes.append("i")
        if n == 2:
            classes.append("sigma(xy)")
            classes.append("sigma(xz)")
            classes.append("sigma(yz)")
        else:
            for i in range(1,n>>1): # = 1:n>>1-1:
                a = i+(n>>1)
                if a > (n>>1):
                    a = n - a
                r,s = reduce(n, a)
                c_class_string(classes, "2S", r, s)
            classes.append("sigma_h")
            if n % 4 == 0:
                classes.append(f"{n>>1}sigma_v")
                classes.append(f"{n>>1}sigma_d")
            else:
                classes.append(f"{n>>1}sigma_d")
                classes.append(f"{n>>1}sigma_v")
        newnames = []
        for i in range(len(names)): # = 1:length(names):
            newnames.append(names[i]+"u")
            names[i] = names[i]+"g"
        names += newnames
        dncharsi = -1 * dnchars
        top = np.hstack((dnchars, dnchars))
        bot = np.hstack((dnchars, dncharsi))
        chars = np.vstack((top, bot))
    else:
        classes.append("sigma_h")
        for i in range(1,(n>>1)+1): # = 1:n>>1:
            if i % 2 == 0:
                r,s = reduce(n, n-i)
            else:
                r,s = reduce(n ,i)
            c_class_string(classes, "2S", r, s)
        classes.append(f"{n}sigma_v")
        newnames = []
        for i in range(len(names)): # = 1:length(names):
            newnames.append(names[i]+"''")
            names[i] = names[i]+"'"
        names += newnames
        dncharsi = -1 * dnchars
        top = np.hstack((dnchars, dnchars))
        bot = np.hstack((dnchars, dncharsi))
        chars = np.vstack((top, bot))
    return names, classes, chars

def Dnd_irr(n):
    if n % 2 == 0:
        n2 = 2*n
        names, classes, chars = Sn_irr(n2)
        #classes = collect(1:2*n2)
        classes = classes[0:n+1]
        for i in range(1,n): # = 2:n:
            classes[i] = "2"+classes[i]
        classes.append(f"{n}C_2'")
        classes.append(f"{n}sigma_d")
        names[0] = "A1"
        names.insert(1, "A2")
        chars = np.vstack((chars[0,:], chars[0,:], chars[1:,:]))
        for i in range(1,n2): # = 2:n2:
            if i >= n2-i:
                break
            chars = chars[:,[j for j in range(np.shape(chars)[1]) if j != n2-i]] # chars[:,0:-1.!=n2-i+2]
        nirr = n+3
        names[2] = "B1"
        names.insert(3, "B2")
        chars = np.vstack((chars[0:3,:], chars[2,:], chars[3:,:]))
        C2p = np.zeros(nirr)
        sigma_d = np.zeros(nirr)
        C2p[0:4] = np.array([1,-1,1,-1])
        sigma_d[0:4] = np.array([1,-1,-1,1])
        chars = np.hstack((chars, C2p[:,None], sigma_d[:,None]))
    else:
        names, classes, dnchars = Dn_irr(n)
        classes.append("i")
        for i in range(1,(n>>1)+1): # = 1:n>>1:
            r,s = reduce(2*n, 2*i+n)
            if s > n:
                s = 2*n-s
            c_class_string(classes, "2S", r, s)
        classes.append(f"{n}sigma_d")
        newnames = []
        for i in range(len(names)): # = 1:length(names):
            newnames.append(names[i]+"u")
            names[i] = names[i]+"g"
        names += newnames
        dncharsi = -1 * dnchars
        top = np.hstack((dnchars, dnchars))
        bot = np.hstack((dnchars, dncharsi))
        chars = np.vstack((top, bot))
    return names, classes, chars
