import pytest
from molsym.symtext.general_irrep_mats import pg_to_symels
from molsym.symtext.goat import mtable_check, goat_chk
from molsym.symtext.multiplication_table import build_mult_table
from molsym.symtools import issame_axis

from .pgs.Cn import *
from .pgs.Sn import *
from .pgs.Cnh import *
from .pgs.Cnv import *
from .pgs.Dn import *
from .pgs.Dnh import *
from .pgs.Dnd import *
from .pgs.Cubic import *

pgs = [
    "C1","Ci","Cs",
    "C2","C3","C4","C5","C6",
    "S4","S6","S8","S10",
    "C2h","C3h","C4h","C5h","C6h",
    "C2v","C3v","C4v","C5v","C6v",
    "D2","D3","D4","D5","D6",
    "D2h","D3h","D4h","D5h","D6h",
    "D2d","D3d","D4d","D5d","D6d",
    "T", "Td", "Th", "O", "Oh"
]

@pytest.mark.parametrize("pg", pgs)
def test_Symel(pg):
    symels, irreps, irrep_mats = pg_to_symels(pg)
    symels_ans = eval(pg+"s")
    rreps = [symel.symbol for symel in symels]
    rreps_ans = [symel.symbol for symel in symels_ans]
    assert set(rreps) == set(rreps_ans)
    print(pg)
    for symel in symels:
        for symel_a in symels_ans:
            if symel.symbol == symel_a.symbol:
                print(symel.symbol)
                print(symel.vector)
                print(symel_a.vector)
                #print(symel.rrep)
                #print(symel_a.rrep)
                if symel.vector is None:
                    assert symel_a.vector is None
                else:
                    assert issame_axis(symel.vector, symel_a.vector)
                assert np.isclose(symel.rrep, symel_a.rrep).all()
   
    mtable = build_mult_table(symels)
    mchk = True
    for k in irrep_mats:
        mtab_chk = mtable_check(k, irrep_mats[k], mtable)
        if mtab_chk == False:
            mchk = False
    assert mchk
    
    gchk = goat_chk(irrep_mats)
    assert gchk