import pytest
from molsym.symtext.goat import irrep_things

test_set = ["C1", "Cs", "Ci", 
            "C2", "C3", "C4", "C5", "C6",
            "C2h", "C3h", "C4h", "C5h", "C6h",
            "C2v", "C3v", "C4v", "C5v", "C6v",
            "D2", "D3", "D4", "D5", "D6", "D8",
            "D2h", "D3h", "D4h", "D5h", "D6h", "D8h",
            "D2d", "D3d", "D4d", "D5d", "D6d",
            "S4", "S6", "S8",
            "T", "Th", "Td", "O", "Oh"]

@pytest.mark.parametrize("irrmat", test_set)
def test_irrep_mats(irrmat):
    print(irrmat)
    got, mtab = irrep_things(irrmat)
    assert got
    assert mtab

@pytest.mark.slow
def test_irrep_mats_Ih():
    print("Ih")
    got, mtab = irrep_things("Ih")
    assert got
    assert mtab
