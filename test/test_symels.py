import pytest
from pgs.Cn import *
from pgs.Cnh import *
from pgs.Cnv import *
from pgs.Dn import *
from pgs.Dnh import *
from pgs.Dnd import *
from pgs.Sn import *
from molsym.symtext.symel import pg_to_symels

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
