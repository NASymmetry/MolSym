"""
Tests for deriving nonstandard-frame irrep matrices
(Symtext.nonstandard_symtext and molsym.salcs.nonstandard_frame).

The intertwiner method: rotate a known-correct standard-frame SALC into the
nonstandard frame by the exact orthogonal real spherical harmonic rotation
matrix for the given angular momentum l (see sh_rep), then solve
D'(g) = Phi'.T T'(g) Phi' directly for the operation matrices of the rotated
frame (see select_dprime_partner_sets). These tests check the properties
that make the result trustworthy: that it's a genuine representation of the
group (satisfies the multiplication table, is orthogonal), that its
characters match the frame-invariant character table, and that the
l-escalation/candidate-rejection logic actually recovers from a bad
candidate rather than silently returning something wrong.
"""
import numpy as np
import pytest
from copy import deepcopy
from pathlib import Path

import molsym
from molsym.salcs.spherical_harmonics import SphericalHarmonicFunctions
from molsym.salcs.projection_op import ProjectionOp

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


def _rotate(standard_symtext):
    """Build a nonstandard_symtext by hand, without going through
    derive_nonstandard_irrep_mats -- used by tests that need to poke at
    select_dprime_partner_sets directly."""
    nonstandard_symtext = deepcopy(standard_symtext)
    Q = standard_symtext.reverse_rotate
    nonstandard_symtext.mol = standard_symtext.mol.transform(Q)
    for symel in nonstandard_symtext.symels:
        R_std = np.array(symel.rrep, dtype=float)
        symel.rrep = Q @ R_std @ Q.T
    return nonstandard_symtext


SYMTEXT_BUILDERS = {
    "ammonia (C3v, real 2D E)": _ammonia_symtext,
    "methane (Td, real 3D T1/T2)": _methane_symtext,
    "boric acid (C3h, complex point group)": _boric_acid_symtext,
}

SYMTEXT_XYZ_FILES = {
    "ammonia (C3v, real 2D E)": "ammonia_rotated.xyz",
    "methane (Td, real 3D T1/T2)": "methane_rotated.xyz",
    "boric acid (C3h, complex point group)": "boric_acid_rotated.xyz",
}

# subgroup -> whether it has two independent, symmetry-defined axes (a real
# paxis and saxis), so its reoriented geometry is fully pinned down and must
# be reproducible across frames. Single-axis groups (Cs, C3, S4, C1, Ci) get
# an arbitrary secondary axis from rotate_mol_to_symels's fallback (crossing
# a *lab-frame* basis vector with paxis: see rotate_mol_to_symels), which
# legitimately differs depending on which direction paxis itself points --
# so only their abstract structure (point group, order, atom_map), not their
# absolute orientation, is expected to match between frames.
SUBGROUPS_TO_TEST = {
    "ammonia (C3v, real 2D E)": {"C3v": True, "Cs": False, "C3": False},
    "methane (Td, real 3D T1/T2)": {"D2d": True, "C3v": True, "S4": False, "C2v": True},
    "boric acid (C3h, complex point group)": {"C3": False, "Cs": False},
}


@pytest.mark.parametrize("label", SUBGROUPS_TO_TEST)
def test_subgroup_symtext_from_nonstandard_frame_matches_standard(label):
    # subgroup_symtext -> subgroup_axes reads symel.vector to pick the new
    # frame's principal/secondary axes. If nonstandard_symtext left those
    # stale (still describing the standard frame) while rotating symel.rrep
    # and mol, subgroup_axes would hand rotate_mol_to_symels axes expressed
    # in the wrong frame, silently reorienting the subgroup molecule
    # inconsistently -- this checks the subgroup derived from a nonstandard
    # symtext lands in exactly the same place as the one from its standard
    # counterpart.
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)

    for subgroup_str, fully_pinned in SUBGROUPS_TO_TEST[label].items():
        sub_std = standard_symtext.subgroup_symtext(subgroup_str)
        sub_nonstd = nonstandard_symtext.subgroup_symtext(subgroup_str)

        assert sub_nonstd.pg.str == sub_std.pg.str == subgroup_str
        assert sub_nonstd.order == sub_std.order
        assert sub_nonstd.complex == sub_std.complex
        np.testing.assert_array_equal(sub_nonstd.atom_map, sub_std.atom_map)
        if fully_pinned:
            np.testing.assert_allclose(sub_nonstd.mol.coords, sub_std.mol.coords, atol=1e-6)


def test_largest_D2h_subgroup_from_nonstandard_frame_matches_standard():
    standard_symtext = _methane_symtext()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)

    sub_std = standard_symtext.largest_D2h_subgroup()
    sub_nonstd = nonstandard_symtext.largest_D2h_subgroup()

    assert sub_nonstd.pg.str == sub_std.pg.str
    assert sub_nonstd.order == sub_std.order
    np.testing.assert_array_equal(sub_nonstd.atom_map, sub_std.atom_map)
    np.testing.assert_allclose(sub_nonstd.mol.coords, sub_std.mol.coords, atol=1e-6)


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_assign_dipole_irrep_is_recomputed_not_stale(label):
    # assign_dipole_irrep is computed once in __init__ from the standard-frame
    # symels/irrep_mats. deepcopy carries that cached value over verbatim, so
    # if nonstandard_symtext doesn't explicitly recompute it after rotating
    # symels and re-deriving irrep_mats, it silently reports which Cartesian
    # axis each irrep's dipole component sits on using stale, standard-frame
    # axes -- wrong for the (generally arbitrarily oriented) nonstandard frame.
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)

    assert nonstandard_symtext.assign_dipole_irrep == nonstandard_symtext.dipole_components_to_irrep()


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_dipole_active_irreps_is_frame_independent(label):
    # dipole_active_irreps only uses symel.rrep traces (basis-independent),
    # unlike dipole_components_to_irrep's exact-raw-axis-alignment check,
    # which reports no activity at all for a generic (nonstandard) orientation
    # even when an irrep genuinely is IR-active -- this is what
    # assemble_dipder_from_dipoles's degeneracy-exploitation path (psi4
    # driver_findif.py) should gate on instead, so it still engages under a
    # nonstandard symtext.
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)

    std_active = standard_symtext.dipole_active_irreps()
    nonstd_active = nonstandard_symtext.dipole_active_irreps()

    assert std_active == nonstd_active
    assert len(std_active) > 0


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_is_nonstandard_flag(label):
    standard_symtext = SYMTEXT_BUILDERS[label]()
    assert standard_symtext.is_nonstandard is False
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    assert nonstandard_symtext.is_nonstandard is True


def test_nonstandard_symtext_rejects_already_nonstandard_input():
    # nonstandard_symtext keeps the original reverse_rotate on its output, so
    # calling it again on that output would apply Q a second time (landing
    # in a Q^2 frame) instead of raising -- is_nonstandard is exactly the
    # flag needed to catch this misuse.
    standard_symtext = _ammonia_symtext()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    with pytest.raises(ValueError):
        molsym.Symtext.nonstandard_symtext(nonstandard_symtext, max_degree=6)


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_nonstandard_symtext_mol_matches_original_geometry(label):
    # The other tests here (multiplication table, orthogonality, characters,
    # subgroup reconstruction) are all self-consistency or frame-invariant
    # checks: they'd still pass even if reverse_rotate were applied with a
    # transposed or sign-flipped Q, since everything downstream (symels,
    # irrep_mats) is rotated by that same wrong Q consistently. The only way
    # to catch that class of error is to compare against geometry that never
    # went through nonstandard_symtext at all -- the original, COM-centered
    # input coordinates.
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)

    reference_mol = molsym.Molecule.from_file(TEST_DIR / "xyz" / SYMTEXT_XYZ_FILES[label])
    reference_mol.translate(reference_mol.find_com())

    assert (nonstandard_symtext.mol.atoms == reference_mol.atoms).all()
    np.testing.assert_allclose(nonstandard_symtext.mol.coords, reference_mol.coords, atol=1e-6)


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_irrep_mats_satisfy_group_multiplication_table(label):
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    h = len(nonstandard_symtext.symels)
    mult_table = nonstandard_symtext.mult_table
    for irrep in nonstandard_symtext.irreps:
        D = [np.array(m) for m in nonstandard_symtext.irrep_mats[irrep.symbol]]
        for gi in range(h):
            for hi in range(h):
                gh = mult_table[gi, hi]
                np.testing.assert_allclose(
                    D[gi] @ D[hi], D[gh], atol=1e-6,
                    err_msg=f"{label} irrep {irrep.symbol}: D(g)D(h) != D(gh)",
                )


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_irrep_mats_are_orthogonal(label):
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    for irrep in nonstandard_symtext.irreps:
        D = [np.array(m) for m in nonstandard_symtext.irrep_mats[irrep.symbol]]
        for Dg in D:
            np.testing.assert_allclose(
                Dg @ np.conj(Dg).T, np.eye(irrep.d), atol=1e-6,
                err_msg=f"{label} irrep {irrep.symbol}: D(g) not orthogonal",
            )


@pytest.mark.parametrize("label", SYMTEXT_BUILDERS)
def test_irrep_mats_characters_match_standard_frame(label):
    # Characters are frame-invariant -- this is the basis-independent check
    # that the nonstandard matrices are really the same abstract irrep as
    # the standard-frame ones, not just some other valid representation.
    standard_symtext = SYMTEXT_BUILDERS[label]()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    for irrep in standard_symtext.irreps:
        chars_std = [np.trace(np.array(m)) for m in standard_symtext.irrep_mats[irrep.symbol]]
        chars_nonstd = [np.trace(np.array(m)) for m in nonstandard_symtext.irrep_mats[irrep.symbol]]
        np.testing.assert_allclose(
            chars_std, chars_nonstd, atol=1e-6,
            err_msg=f"{label} irrep {irrep.symbol}: characters not frame-invariant",
        )


def test_1d_irreps_are_copied_directly_not_solved():
    # Boric acid's C3h is a complex point group where every irrep is 1D,
    # including the E(1)/E(2) complex-conjugate pairs. 1D irrep matrices
    # are frame-invariant by definition (a 1x1 matrix is unchanged by any
    # similarity transform), so they should come back byte-identical to
    # the standard-frame values rather than going through the SALC solve --
    # which matters because running them through the solve is exactly what
    # used to silently fail for complex point groups: remove_complexity
    # recombines conjugate pairs into real vectors that are individually
    # NOT group-invariant, only together as a 2D pair.
    standard_symtext = _boric_acid_symtext()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    for irrep in standard_symtext.irreps:
        assert irrep.d == 1
        std_mats = np.array(standard_symtext.irrep_mats[irrep.symbol])
        nonstd_mats = np.array(nonstandard_symtext.irrep_mats[irrep.symbol])
        np.testing.assert_allclose(std_mats, nonstd_mats, atol=1e-8)


def test_missing_irrep_raises_runtime_error():
    from molsym.salcs.nonstandard_frame import derive_nonstandard_irrep_mats
    # Methane's T_1 doesn't appear in the polynomial basis until degree 3
    # (translations are T_2 at degree 1; T_1 needs a higher-degree block).
    # max_degree=2 should fail loudly instead of silently returning a dict
    # that's missing T_1.
    standard_symtext = _methane_symtext()
    nonstandard_symtext = _rotate(standard_symtext)
    with pytest.raises(RuntimeError):
        derive_nonstandard_irrep_mats(standard_symtext, nonstandard_symtext, max_degree=2)


def test_multiplicity_recovers_from_a_bad_candidate(monkeypatch):
    # Ammonia's E irrep resolves at l=1, where there's only a single
    # candidate partner set -- so this poisons that sole candidate to be
    # rank-deficient and checks that the real selection code rejects it and
    # escalates to l=2 (where E has multiplicity 2, i.e. two genuine
    # candidates) rather than silently accepting something broken.
    import molsym.salcs.nonstandard_frame as nf
    from molsym.salcs.salc import SALCs

    standard_symtext = _ammonia_symtext()
    nonstandard_symtext = _rotate(standard_symtext)

    real_salc_to_phi = SALCs.salc_to_phi
    poisoned = {"done": False}

    def poisoned_salc_to_phi(salcs, nfxn):
        Phi = real_salc_to_phi(salcs, nfxn)
        rows = salcs if isinstance(salcs, (list, tuple)) else [salcs]
        if rows[0].irrep.symbol == "E" and not poisoned["done"]:
            poisoned["done"] = True
            Phi = Phi.copy()
            Phi[:, 1] = Phi[:, 0]  # duplicate a column -> rank-deficient
        return Phi

    monkeypatch.setattr(SALCs, "salc_to_phi", staticmethod(poisoned_salc_to_phi))

    Q = standard_symtext.reverse_rotate
    selected_mats, selected_info = nf.select_dprime_partner_sets(
        standard_symtext, nonstandard_symtext, Q, max_l=5,
        sh_function_cls=SphericalHarmonicFunctions, projection_op_cls=ProjectionOp,
    )

    assert selected_info["E"]["l"] == 2  # escalated past the poisoned l=1 candidate

    D = [np.array(m) for m in selected_mats["E"]]
    chars_recovered = [np.trace(m) for m in D]
    chars_std = [np.trace(np.array(m)) for m in standard_symtext.irrep_mats["E"]]
    np.testing.assert_allclose(chars_recovered, chars_std, atol=1e-6)


def test_agrees_with_schur_lemma_method():
    # Cross-check against the independent Schur's-lemma/random-operator
    # derivation (rotate_irrmats): two structurally unrelated derivations
    # landing on the same characters is much stronger evidence than either
    # one alone. Skipped if that module isn't present -- it's WIP and may
    # move to a separate branch rather than ship alongside this method.
    rotate_irrep_mats = pytest.importorskip("molsym.salcs.rotate_irrep_mats")
    standard_symtext = _ammonia_symtext()
    nonstandard_symtext = molsym.Symtext.nonstandard_symtext(standard_symtext, max_degree=6)
    D_schur = rotate_irrep_mats.rotate_irrmats(standard_symtext, nonstandard_symtext)
    for irrep in standard_symtext.irreps:
        chars_pinv = [np.trace(np.array(m)) for m in nonstandard_symtext.irrep_mats[irrep.symbol]]
        chars_schur = [np.trace(np.array(m)) for m in D_schur[irrep.symbol]]
        np.testing.assert_allclose(chars_pinv, chars_schur, atol=1e-6)
