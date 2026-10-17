Usage
=====

MolSym is an open-source Python package for handling molecular symmetry.
Features of MolSym include:

*   Point group detection
*   Molecule symmetrization
*   Symmetry element, irreducible representation, irreducible representation matrices, and character table generation for arbitrary point groups
*   Symmetry adapted linear combinations for internal coordinates, Cartesian nuclear coordinates, spherical harmonic basis functions, and Cartesian basis functions

The examples below assume ``water.xyz`` contains a bent water geometry, and:

.. code-block:: python

    import molsym

Point Group detection
---------------------

``molsym.find_point_group`` detects the Schoenflies point group of a
``molsym.Molecule``. Detection works up to a numerical tolerance
(``mol.tol``, ``1e-5`` by default) on how far the input geometry may
deviate from an exact symmetry element, so it also works on molecules
that are only approximately symmetric, such as the result of a geometry
optimization:

.. code-block:: python

    mol = molsym.Molecule.from_file("water.xyz")
    pg, (paxis, saxis) = molsym.find_point_group(mol)
    # pg == "C2v"

``find_point_group`` returns the point group as a string, along with a
primary and secondary axis that are used to orient the molecule
consistently with respect to its symmetry elements.

Symmetrization
--------------

Real geometries (e.g. from an optimization) are rarely *exactly*
symmetric. ``molsym.symmetrize`` detects a molecule's point group and
projects its atoms exactly onto that point group's symmetry elements,
returning a new, exactly-symmetric ``molsym.Molecule``:

.. code-block:: python

    mol = molsym.Molecule.from_file("water.xyz")
    mol = molsym.symmetrize(mol, asym_tol=0.05)

``asym_tol`` (default ``0.05``) is the tolerance used to detect the point
group of the possibly-imperfect input geometry; the returned molecule's
own tolerance is tightened to ``1e-12``. Symmetrizing a molecule before
building a :ref:`Symtext <symtext-generation>` avoids spurious symmetry
breaking caused by numerical noise in the input coordinates.

.. _symtext-generation:

Symtext generation
-------------------

A ``molsym.Symtext`` is MolSym's central data structure: it holds a
molecule's point group, symmetry elements, the mapping of atoms under
each symmetry element, the group multiplication table, and the
irreducible representations (with matrices) of the point group. Every
SALC-construction routine in MolSym operates on a ``Symtext``.

Build one from an already-symmetric ``molsym.Molecule``:

.. code-block:: python

    mol = molsym.symmetrize(molsym.Molecule.from_file("water.xyz"))
    symtext = molsym.Symtext.from_molecule(mol)

or directly from a file, which skips the intermediate ``Molecule`` and
does not symmetrize the geometry first (symmetrize it yourself, as
above, if the input coordinates are only approximately symmetric):

.. code-block:: python

    symtext = molsym.Symtext.from_file("water.xyz")

A ``Symtext`` exposes the point group, symmetry elements, order, and
irreducible representations:

.. code-block:: python

    symtext.pg.str                          # 'C2v'
    [s.symbol for s in symtext.symels]      # ['E', 'C_2', 'sigma_v(0)', 'sigma_d(0)']
    [ir.symbol for ir in symtext.irreps]    # ['A_1', 'A_2', 'B_1', 'B_2']
    symtext.order                           # 4

SALCs
-----

Symmetry-adapted linear combinations (SALCs) are built by projecting a
set of basis functions (a "function set") onto the irreducible
representations of a ``Symtext``, using ``molsym.salcs.ProjectionOp``.
MolSym provides function sets for internal coordinates, Cartesian
nuclear displacements, and spherical harmonic/polynomial basis
functions.

Internal coordinates
~~~~~~~~~~~~~~~~~~~~~

Internal coordinates are specified as a sequence of
``[[atom indices], label]`` pairs, using 0-based atom indices. The
label's first character(s) select the coordinate type: ``R`` for a bond
stretch, ``A`` for a bend, ``D`` for a torsion, ``O`` for an
out-of-plane, and ``Lin``/``Lx``/``Ly`` for linear bends.

.. code-block:: python

    ics = ([[0, 1], "R1"], [[0, 2], "R2"], [[1, 0, 2], "A1"])
    ic_fxn_set = molsym.salcs.InternalCoordinates(symtext, ics)
    ic_salcs = molsym.salcs.ProjectionOp(symtext, ic_fxn_set)
    print(ic_salcs)

For water's two O-H stretches and H-O-H bend, this produces three
SALCs: the totally-symmetric combination of the two stretches, the
bend (both ``A_1``), and the antisymmetric stretch combination
(``B_2``).

Cartesian displacements and Eckart conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``molsym.salcs.CartesianCoordinates`` builds SALCs of the ``3N``
Cartesian displacement coordinates of all atoms. By default,
``ProjectionOp`` projects out translational and rotational motion (the
Eckart conditions), leaving only the ``3N-6`` (or ``3N-5`` for linear
molecules) vibrational SALCs:

.. code-block:: python

    cart_coords = molsym.salcs.CartesianCoordinates(symtext)
    vib_salcs = molsym.salcs.ProjectionOp(symtext, cart_coords)
    len(vib_salcs)                             # 3 for water (3*3 - 6)
    [s.irrep.symbol for s in vib_salcs.salcs]  # ['A_1', 'A_1', 'B_2']

The ``project_Eckart`` keyword controls which subspace is removed:

.. list-table::
   :header-rows: 1

   * - ``project_Eckart``
     - Effect
   * - ``"both"`` (default)
     - Project out both translations and rotations; only vibrations remain.
   * - ``None``
     - Project out nothing; all ``3N`` Cartesian SALCs are kept.
   * - ``"translational"``
     - Project out only translations; rotations and vibrations remain.
   * - ``"rotational"``
     - Project out only rotations; translations and vibrations remain.

Spherical harmonic and polynomial basis functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``molsym.salcs.SphericalHarmonics`` builds SALCs of atom-centered
spherical harmonic basis functions (e.g. for symmetrizing an
atomic-orbital basis set). Its function set is one list of shell
angular momenta (``l``) per atom:

.. code-block:: python

    # A single s-shell (l=0) on each of water's 3 atoms
    sph_fxn_set = molsym.salcs.SphericalHarmonics(symtext, [[0], [0], [0]])
    sph_salcs = molsym.salcs.ProjectionOp(symtext, sph_fxn_set)

``molsym.salcs.polynomial_functions.PolynomialFunctions`` similarly
builds SALCs of Cartesian polynomial basis functions up to a given
degree centered on each atom:

.. code-block:: python

    from molsym.salcs.polynomial_functions import PolynomialFunctions

    poly_fxn_set = PolynomialFunctions(symtext, degree=1)
    poly_salcs = molsym.salcs.ProjectionOp(symtext, poly_fxn_set, project_Eckart=None)

``project_Eckart`` only has an effect for ``CartesianCoordinates``
function sets, so it is conventionally set to ``None`` for internal
coordinate, spherical harmonic, and polynomial function sets to make
that explicit.

Subgroups
---------

``Symtext.subgroup_symtext`` builds a ``Symtext`` for a subgroup of the
molecule's point group, for example to work in an Abelian group. The
point-group name may be given in any capitalization (``"C2"``, ``"c2"``):

.. code-block:: python

    c2 = symtext.subgroup_symtext("c2")
    c2.pg.str                               # 'C2'
    [ir.symbol for ir in c2.irreps]         # ['A', 'B']

    symtext.largest_D2h_subgroup().pg.str   # 'C2v', the largest subgroup of D2h

Requesting a group the molecule does not have raises an exception.

Symtexts in the input frame
---------------------------

``Symtext.from_molecule`` reorients the molecule into a standard frame for
its point group, so ``symtext.mol`` (and every SALC built from it) is
expressed in that frame rather than in the input coordinates.
``molsym.Symtext.nonstandard_symtext`` returns the same ``Symtext``
rotated back to the input orientation (centered on the center of mass),
with symmetry elements and irreducible representation matrices derived for
that orientation, so SALCs can be used directly with the input geometry:

.. code-block:: python

    symtext = molsym.Symtext.from_molecule(molsym.Molecule.from_file("water.xyz"))
    in_frame = molsym.Symtext.nonstandard_symtext(symtext)
    in_frame.is_nonstandard                 # True
    cart = molsym.salcs.CartesianCoordinates(in_frame)
    vib_salcs = molsym.salcs.ProjectionOp(in_frame, cart)   # displacements in the input frame

A subgroup taken from a standard-frame ``Symtext`` can be returned to the
input frame the same way: ``nonstandard_symtext(symtext.subgroup_symtext("C2"))``.

Linear molecules
----------------

Linear molecules have the infinite point groups C∞v and D∞h. Their
symmetry elements have no explicit matrices, so Cartesian SALCs are built
with ``molsym.salcs.cartesian_coordinates.LinearCartesian`` instead of
``CartesianCoordinates``. The molecule may lie along any axis, and only the
``3N-5`` vibrations remain once translations and rotations are projected
out:

.. code-block:: python

    import numpy as np
    from molsym.salcs.cartesian_coordinates import LinearCartesian

    co2 = molsym.Molecule(["C", "O", "O"],
                          np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
                          [12.0, 15.995, 15.995])
    linear = molsym.Symtext.from_molecule(co2)
    str(linear.pg)                          # 'Dinfh'
    lin_salcs = molsym.salcs.ProjectionOp(linear, LinearCartesian(linear))
    lin_salcs.sort_to("blocks")
    [s.irrep.symbol for s in lin_salcs.salcs]   # ['Sigma_g^+', 'Sigma_u^+', 'Pi_u', 'Pi_u']

A linear ``Symtext`` is already in the input frame, so ``nonstandard_symtext``
is neither needed nor accepted for it. Finite subgroups of a linear group
(``"D2h"``, ``"C2v"``, ``"D4h"``, ...) are oriented with their principal axis
along the molecular axis, and a ``Cs`` mirror contains the axis:

.. code-block:: python

    linear.subgroup_symtext("D2h").pg.str   # 'D2h'

Single atoms
------------

A single atom has the point group Kh. Its three Cartesian displacements
are the components of the ``P`` irreducible representation and are pure
translations, so projecting out translations leaves no SALCs:

.. code-block:: python

    atom = molsym.Symtext.from_molecule(molsym.Molecule(["He"], np.array([[0.0, 0.0, 0.0]]), [4.0026]))
    atom.pg.str                             # 'Kh'
    len(molsym.salcs.ProjectionOp(atom, molsym.salcs.CartesianCoordinates(atom)))   # 0
    p = molsym.salcs.ProjectionOp(atom, molsym.salcs.CartesianCoordinates(atom), project_Eckart=None)
    [s.irrep.symbol for s in p.salcs]       # ['P', 'P', 'P']

Helpers for finite differences
------------------------------

``molsym.salcs.salc_tools`` provides the symmetry operations a
finite-difference code needs to skip displacements, for Cartesian SALCs
of any point group, including linear groups:

* ``maps_to_negative(symtext, salc)`` is ``True`` when some symmetry
  operation turns the displacement into its negative, so the + and -
  displacements are equivalent and only one needs to be computed.
* ``generate_symmetric_partner(symtext, salc, data, data_type)`` generates
  a dipole (``data_type="dipole"``) or gradient (``"gradient"``) at the +
  displacement from the one computed at the - displacement.
* ``generate_degenerate_partner(symtext, salc, partner_salc, data, data_type)``
  generates the same quantity for another component of a degenerate
  linear-group (Pi) SALC.

.. code-block:: python

    from molsym.salcs.salc_tools import (maps_to_negative,
        generate_symmetric_partner, generate_degenerate_partner)

    vib_salcs.sort_to("blocks")
    [maps_to_negative(in_frame, s) for s in vib_salcs.salcs]    # [False, False, True]
    neg_gradient = np.zeros((3, 3))         # gradient at the - displacement, one row per atom
    pos_gradient, op = generate_symmetric_partner(in_frame, vib_salcs.salcs[2], neg_gradient, data_type="gradient")

    pi_x, pi_y = lin_salcs.salcs[2], lin_salcs.salcs[3]
    generate_degenerate_partner(linear, pi_x, pi_y, np.array([0.0, 1.0, 0.0]))   # dipole for pi_y, +/- z
