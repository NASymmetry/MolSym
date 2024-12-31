import pytest
import numpy as np
from molsym.symtext.character_table import CharacterTable, pg_to_chartab
from pgs.Cn import *
from pgs.Cnh import *
from pgs.Cnv import *
from pgs.Dn import *
from pgs.Dnh import *
from pgs.Dnd import *
from pgs.Sn import *
from test_symels import pgs

@pytest.mark.parametrize("pg", pgs)
def test_CharTable(pg):
    ctab = pg_to_chartab(pg)
    ctab_ans = CharacterTable(pg,np.array(eval(pg+"irr")),np.array(eval(pg+"cn")),None,eval(pg+"ct"),None)
    beans = ctab == ctab_ans
    if not beans:
        print(ctab)
        print("Ref.")
        print(ctab_ans)
        tab_chk = np.isclose(ctab.characters, ctab_ans.characters, atol=1e-14)
        irr_chk = ctab.irreps == ctab_ans.irreps
        name_chk = ctab.classes == ctab_ans.classes
        print(f"Table Check: {tab_chk.all()}")
        print(f"Irrep. Check: {irr_chk.all()}")
        print(f"Name Check: {name_chk.all()}")
        if not tab_chk.all():
            print(tab_chk)
            print(ctab.characters-ctab_ans.characters)
        elif not name_chk.all():
            print(name_chk)
    assert beans
