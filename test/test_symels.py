import os
import pytest
import numpy as np
from pgs.Cn import *
from pgs.Cnh import *
from pgs.Cnv import *
from pgs.Dn import *
from pgs.Dnh import *
from pgs.Dnd import *
from pgs.Sn import *
from molsym.symtext.main import pg_to_symels, pg_to_chartab, cn_class_map, generate_symel_to_class_map
from molsym import Symtext

PATH = os.path.dirname(os.path.realpath(__file__))

pgs = [
    "C2","C3","C4","C5","C6",
    "C2h","C3h","C4h","C5h","C6h",
    "C2v","C3v","C4v","C5v","C6v",
    "S4","S6","S8","S10",
    "D2","D3","D4","D5","D6",
    "D2d","D3d","D4d","D5d","D6d",
    "D2h","D3h","D4h","D5h","D6h"]

@pytest.mark.parametrize("pg", pgs)
def test_Symel(pg):
    symels = pg_to_symels(pg)
    symels_ans = eval(pg+"s")
    beans = True
    slen = len(symels)
    slen_ans = len(symels_ans)
    if slen != slen_ans:
        beans = False
    for i in range(slen):
        if symels[i] == symels_ans[i]:
            continue
        else:
            print(f"{pg} Symels {symels[i]} and {symels_ans[i]} do not match!")
            print("Calculated:")
            print(symels)
            print("Ref:")
            print(symels_ans)
            beans = False
    assert beans

def test_symtext():
    #for i in range(2,7):
    #    pg = f"D{i}h"
    #    beans = generate_symel_to_class_map(pg_to_symels(pg),pg_to_chartab(pg))
    #    print(f"pg: {pg}, {beans}")
    file_path = os.path.join(PATH, "sxyz", "D6h.xyz")
    print(Symtext.from_file(file_path))
