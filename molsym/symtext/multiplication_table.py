import numpy as np
from .symtext import *

def multifly(symels, A, B):
    Crrep = np.dot(A.rrep,B.rrep)
    for (i,g) in enumerate(symels):
        if np.isclose(Crrep, g.rrep).all():
            return i,g
    raise Exception(f"No match found for Symels {A.symbol} and {B.symbol}!")

def build_mult_table(symels):
    h = len(symels)
    t = np.zeros((h,h))
    for (i,a) in enumerate(symels):
        for (j,b) in enumerate(symels):
            t[i,j] = multifly(symels, a, b)[0]
    return t