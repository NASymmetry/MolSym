import numpy as np
from molsym.symtext.symtext import CharTable
from pgs.Cn import *
from pgs.Cnh import *
from pgs.Cnv import *
from pgs.Dn import *
from pgs.Dnh import *
from pgs.Dnd import *
from pgs.Sn import *
from molsym.symtext.main import pg_to_chartab
from test_symels import pgs


def test_CharTable():
    for pg in pgs:
        ctab_a = pg_to_chartab(pg)
        ctab_b = CharTable(pg,np.array(eval(pg+"irr")),np.array(eval(pg+"cn")),None,eval(pg+"ct"),None)
        beans = ctab_a == ctab_b
        if not beans:
            print(ctab_a)
            print("Ref.")
            print(ctab_b)
            tab_chk = ctab_a.characters == ctab_b.characters
            irr_chk = ctab_a.irreps == ctab_b.irreps
            name_chk = ctab_a.classes == ctab_b.classes
            print(f"Table Check: {tab_chk.all()}")
            print(f"Irrep. Check: {irr_chk.all()}")
            print(f"Name Check: {name_chk.all()}")
            if not tab_chk.all():
                print(tab_chk)
                print(ctab_a.characters-ctab_b.characters)
            elif not name_chk.all():
                print(name_chk)
        assert beans
