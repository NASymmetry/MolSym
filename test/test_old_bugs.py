"""
    Put old bugged cases here so we can make sure we don't reintroduce them!
"""
import os
import numpy as np
import molsym

PATH = os.path.dirname(os.path.realpath(__file__))
# Test formaldehyde. Bug was planar molecules not being assigned a secondary axis
def test_formaldehyde():
    file_path = os.path.join(PATH, "xyz", f"formaldehyde.xyz")
    mol = molsym.Molecule.from_file(file_path)
    pg, (paxis, saxis) = molsym.find_point_group(mol)

    assert pg == "C2v"
    x,y,z = np.eye(3)
    assert (np.isclose(paxis, z).all() or np.isclose(paxis, -z).all())
    assert (np.isclose(saxis, x).all() or np.isclose(saxis, -x).all())
