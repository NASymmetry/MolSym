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
   * - ``True`` (default)
     - Project out both translations and rotations; only vibrations remain.
   * - ``False``
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
    poly_salcs = molsym.salcs.ProjectionOp(symtext, poly_fxn_set, project_Eckart=False)

``project_Eckart`` only has an effect for ``CartesianCoordinates``
function sets, so it is conventionally set to ``False`` for internal
coordinate, spherical harmonic, and polynomial function sets to make
that explicit.
