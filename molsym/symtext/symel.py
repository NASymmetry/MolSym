import re
from dataclasses import dataclass
import numpy as np

class PointGroup():
    def __init__(self, s, family, n, subfamily):
        self.str = s
        self.family = family
        self.n = n
        self.subfamily = subfamily
    
    @classmethod
    def from_string(cls, s):
        regex = r"([A-Z]+)(\d+)?([a-z]+)?"
        m = re.match(regex, s)
        family, n, subfamily = m.groups()
        if n is not None:
            n = int(n)
        if subfamily is not None:
            subfamily = str(subfamily)
        family = str(family)
        return cls(s, family, n, subfamily)

    def __str__(self):
        return "Family: "+self.family+"\nn: "+str(self.n)+"\nSubfamily: "+self.subfamily

@dataclass
class Symel():
    symbol:str
    rrep:np.array
    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True, formatter={"all":lambda x: f"{x:8.5f}"}):
            return f"\nSymbol: {self.symbol:>10s}: [{self.rrep[0,:]},{self.rrep[1,:]},{self.rrep[2,:]}]"
    def __repr__(self) -> str:
        return self.__str__()
    def __eq__(self, other):
        return self.symbol == other.symbol and np.isclose(self.rrep,other.rrep,atol=1e-10).all()

class CharTable():
    def __init__(self, pg, irreps, classes, class_orders, chars, irrep_dims) -> None:
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

def reduce(n, i):
    g = gcd(n, i)
    return n//g, i//g # floor divide to get an int, there should never be a remainder since we are dividing by the gcd

def gcd(A, B):
    # A quick implementation of the Euclid algorithm for finding the greatest common divisor
    a = max(A,B)
    b = min(A,B)
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        r = a % b
        return gcd(b, r)

def divisors(n):
    # This isn't meant to handle large numbers, thankfully most point groups have an order less than 100
    out = []
    for i in range(n):
        if n % (i+1) == 0:
            out.append(i+1)
    return out