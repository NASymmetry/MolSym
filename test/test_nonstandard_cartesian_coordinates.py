"""
Tests for CartesianCoordinates' dense-frame equivalence-set fix
(get_symmetry_equiv_functions_nonstandard), auto-dispatched via
symtext.is_nonstandard.

get_symmetry_equiv_functions() (left untouched) decides which Cartesian
displacement coordinates need their own SALC-projection seed by checking
which entries of the group's operation matrices are nonzero, using exactly
one seed per merged set with no check that one seed actually suffices. That
assumption only holds when the matrices are sparse -- true in the standard
orientation but not in a genuinely rotated (nonstandard) frame, where it
silently undercounts SALCs. These tests check that the nonstandard method
recovers the correct count, and that the standard/legacy path is unaffected.
"""
import numpy as np
import pytest
from pathlib import Path

import molsym

TEST_DIR = Path(__file__).resolve().parent


def _ammonia_symtext():
    mol = molsym.Molecule.from_file(TEST_DIR / "xyz" / "ammonia_rotated.xyz")
    return molsym.Symtext.from_molecule(mol)


def _methane_symtext():
    mol = molsym.Molecule.from_file(TEST_DIR / "xyz" / "methane_rotated.xyz")
    return molsym.Symtext.from_molecule(mol)


def _boric_acid_symtext():
    mol = molsym.Molecule.from_file(TEST_DIR / "xyz" / "boric_acid_rotated.xyz")
    return molsym.Symtext.from_molecule(mol)


SYMTEXT_BUILDERS = {
    "ammonia": _ammonia_symtext,
    "methane": _methane_symtext,
    "boric_acid": _boric_acid_symtext,
}


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_standard_frame_agrees_with_legacy_method(label):
    # On an already-standard (sparse) frame, the two methods should pick
    # the same representative seeds -- the fix should never change behavior
    # there, only in the dense/nonstandard case.
    standard_symtext = SYMTEXT_BUILDERS[label]()
    cart = molsym.salcs.CartesianCoordinates(standard_symtext)
    legacy_seeds = sorted(min(s) for s in cart.get_symmetry_equiv_functions())
    nonstandard_seeds = sorted(s[0] for s in cart.get_symmetry_equiv_functions_nonstandard())
    assert legacy_seeds == nonstandard_seeds


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_dense_frame_salc_count_matches_standard_frame(label):
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    assert nonstandard_symtext.is_nonstandard

    cart_std = molsym.salcs.CartesianCoordinates(standard_symtext)
    cart_nonstd = molsym.salcs.CartesianCoordinates(nonstandard_symtext)

    salcs_std = molsym.salcs.ProjectionOp(standard_symtext, cart_std, project_Eckart=None)
    salcs_nonstd = molsym.salcs.ProjectionOp(nonstandard_symtext, cart_nonstd, project_Eckart=None)

    natom = len(standard_symtext.mol)
    assert len(salcs_std.salcs) == 3 * natom
    assert len(salcs_nonstd.salcs) == 3 * natom, (
        f"{label}: nonstandard frame undercounted SALCs "
        f"({len(salcs_nonstd.salcs)} vs expected {3 * natom}) -- "
        "this is the bug get_symmetry_equiv_functions_nonstandard fixes"
    )


def test_boric_acid_random_rotation_is_the_case_that_exposed_the_bug():
    # This specific geometry (a Haar-random rotation, not an axis-preserving
    # one) is what originally surfaced the bug: the legacy method
    # undercounted 15 SALCs instead of 21 here, because boric acid's own
    # C3h axis-preserving symmetry made every earlier test case accidentally
    # keep symel.rrep sparse even after "rotation". Kept as its own explicit
    # regression test since it's the concrete case the fix was written for.
    standard_symtext = _boric_acid_symtext()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)

    cart_nonstd = molsym.salcs.CartesianCoordinates(nonstandard_symtext)
    salcs_nonstd = molsym.salcs.ProjectionOp(nonstandard_symtext, cart_nonstd, project_Eckart=None)
    assert len(salcs_nonstd.salcs) == 21


def test_no_duplicate_seeds_in_nonstandard_equiv_functions():
    # Regression test for a real bug introduced (and fixed) during
    # development: an earlier version of get_symmetry_equiv_functions_nonstandard
    # processed overlapping-but-distinct orbits independently instead of
    # merging them first (via a frontier expansion), which could emit the
    # same seed coordinate twice.
    standard_symtext = _boric_acid_symtext()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    cart = molsym.salcs.CartesianCoordinates(nonstandard_symtext)
    seeds = [s[0] for s in cart.SE_fxns]
    assert len(seeds) == len(set(seeds)), f"duplicate seeds found: {seeds}"


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_with_eckart_projection_dof_count(label):
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    cart_nonstd = molsym.salcs.CartesianCoordinates(nonstandard_symtext)
    salcs_nonstd = molsym.salcs.ProjectionOp(nonstandard_symtext, cart_nonstd, project_Eckart="both")
    natom = len(standard_symtext.mol)
    assert len(salcs_nonstd.salcs) == 3 * natom - 6
