==================
OpenMC Integration
==================

UM2 provides a workflow for converting supported models created with
the real OpenMC Python API into unstructured meshes and MPACT-compatible
input files.

This workflow combines continuous-energy OpenMC calculations, multigroup
cross-section generation, geometry conversion, MPACT hierarchy inference,
and UM2 mesh generation.

Unlike the :doc:`quickstart` tutorial, this workflow uses the real
``openmc`` package rather than ``um2.openmc``.

Overview
--------

The conversion workflow consists of the following stages:

1. Construct a continuous-energy model using OpenMC.
2. Generate material-wise multigroup cross sections with OpenMC,
   or provide an existing compatible MGXS library.
3. Convert the OpenMC geometry and materials into UM2 objects.
4. Infer the MPACT spatial hierarchy.
5. Generate an unstructured mesh using UM2.
6. Export the mesh, material definitions, cross sections, and
   MPACT input files.

Model generation and file export are separate operations. This allows
the generated model to be inspected before it is written to disk.

Prerequisites
-------------

Before starting, install the UM2 Python API as described in
:doc:`installation`.

The complete workflow also requires:

* The real OpenMC Python package.
* A working OpenMC executable.
* Continuous-energy nuclear data compatible with OpenMC.
* The compiled UM2 Python bindings.
* Gmsh and its shared library.

Set the ``OPENMC_CROSS_SECTIONS`` environment variable to the location
of an OpenMC continuous-energy cross-section index file:

.. code-block:: bash

   export OPENMC_CROSS_SECTIONS="/path/to/cross_sections.xml"

Verify that OpenMC is available:

.. code-block:: bash

   openmc --version

If OpenMC is installed outside the system PATH, its executable can be
specified using the ``openmc_exec`` argument.

Creating an OpenMC Model
------------------------

The following example constructs a two-dimensional pin-cell geometry
using a real OpenMC model.

It consists of a cylindrical uranium dioxide fuel region surrounded
by water, inside a square reflective boundary.

Create a Python file named ``openmc_to_um2.py``:

.. code-block:: python

   from pathlib import Path

   import openmc
   import um2

   from um2.generate_um2_from_openmc import generate_um2_from_openmc
   from um2.pipeline import write_mesh_result

   # Define continuous-energy materials.
   fuel = openmc.Material(name="fuel")
   fuel.add_element("U", 1.0, enrichment=3.2)
   fuel.add_element("O", 2.0)
   fuel.set_density("g/cm3", 10.4)

   water = openmc.Material(name="water")
   water.add_element("H", 2.0)
   water.add_element("O", 1.0)
   water.set_density("g/cm3", 0.74)

   materials = openmc.Materials([fuel, water])

   # Construct the pin-cell geometry.
   fuel_boundary = openmc.ZCylinder(r=0.44)

   outer_boundary = openmc.model.RectangularPrism(
       1.26, 1.26, boundary_type="reflective"
   )

   fuel_cell = openmc.Cell(
       fill=fuel,
       region=-fuel_boundary
   )

   water_cell = openmc.Cell(
       fill=water,
       region=+fuel_boundary & -outer_boundary
   )

   root = openmc.Universe(cells=[fuel_cell, water_cell])
   geometry = openmc.Geometry(root)

   # Configure a continuous-energy OpenMC calculation.
   settings = openmc.Settings()
   settings.run_mode = "eigenvalue"
   settings.batches = 50
   settings.inactive = 10
   settings.particles = 1000

   settings.source = openmc.IndependentSource(
       space=openmc.stats.Point((0.0, 0.0, 0.0))
   )

   model = openmc.Model(
       geometry=geometry,
       materials=materials,
       settings=settings
   )

   # Initialize UM2 and Gmsh.
   um2.initialize()

   try:
       # Generate MGXS, convert geometry, and create the mesh.
       result = generate_um2_from_openmc(
           model,
           energy_groups="CASMO-8",
           kn_target=12.0,
           mesh_type="Tri",
           workdir="output/openmc",
       )

       # Export the generated MPACT model.
       output = Path("output/pincell.xdmf")
       files = write_mesh_result(result, output)

       print("Generated files:")
       for file in files:
           print(file)

   finally:
       um2.finalize()

The example uses a small number of particles and batches for
demonstration. Production cross sections require appropriate
statistical convergence and verification.

Generating Multigroup Cross Sections
------------------------------------

When ``mgxs_path`` is not provided, ``generate_um2_from_openmc()``
automatically calls ``generate_mgxs()``.

This function creates material-wise multigroup cross-section tallies,
runs OpenMC, processes the resulting statepoint, and exports a
multigroup HDF5 library.

The original OpenMC model is deep-copied before modifying its tallies
and simulation settings.

The generated library contains macroscopic cross sections associated
with individual materials.

By default, the integration uses the ``CASMO-8`` energy-group structure.

The automatic MGXS workflow requires a working continuous-energy
OpenMC calculation and the corresponding nuclear data.

Using an Existing MGXS Library
------------------------------

If a compatible multigroup library has already been generated, it can
be supplied directly to avoid rerunning OpenMC.

For example:

.. code-block:: python

   result = generate_um2_from_openmc(
       model,
       mgxs_path="output/openmc/mgxs.h5",
       kn_target=12.0,
       mesh_type="Tri",
   )

The supplied library must correspond to the model's materials and
contain the cross-section datasets expected by the converter.

This mode still performs geometry conversion and UM2 mesh generation,
but skips automatic MGXS generation.

Generating the UM2 Mesh
-----------------------

After obtaining the multigroup library, the integration converts
supported OpenMC surfaces, regions, cells, materials, universes,
and lattices into UM2-compatible representations.

UM2 then infers the MPACT spatial hierarchy:

* Core
* Assemblies
* Lattices
* Ray tracing modules (RTMs)
* Coarse cells

The model is meshed with UM2 using Gmsh.

The ``kn_target`` argument controls Knudsen-number-based mesh sizing
using the multigroup cross sections.

The ``mesh_type`` argument selects the element type, such as
``Tri`` or ``QuadraticTri``.

The function returns a ``MeshResult`` object containing the
generated model, inferred partition, material information,
and cross-section metadata.

It does not automatically write the final MPACT model files.

Exporting the MPACT Model
-------------------------

Use ``write_mesh_result()`` to export a previously generated model:

.. code-block:: python

   from um2.pipeline import write_mesh_result

   files = write_mesh_result(
       result,
       "output/pincell.xdmf",
   )

The export process produces files including:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extension
     - Description
   * - ``.xdmf``
     - UM2 model geometry and mesh metadata.
   * - ``.h5``
     - HDF5 data associated with the XDMF model.
   * - ``.input``
     - MPACT input file, including material mappings and boundary conditions.
   * - ``.xsl``
     - MPACT USER multigroup cross-section library.
   * - ``.volumes``
     - Material volume information from the mesh.
   * - ``.xs``
     - Additional UM2 material cross-section output.

The generated ``.xsl`` is distinct from the OpenMC ``mgxs.h5``
library used during model generation.

The export process maps OpenMC material cross-section datasets
to the material definitions expected by MPACT.

Boundary Conditions
-------------------

For supported rectangular outer boundaries, the integration
transfers OpenMC boundary conditions into the MPACT input file.

The six boundary-condition values are ordered as follows:

.. code-block:: text

   west north east south top bottom

The current mapping is:

.. list-table::
   :header-rows: 1

   * - OpenMC boundary type
     - MPACT value
   * - ``vacuum``
     - ``0``
   * - ``reflective``
     - ``1``
   * - ``periodic``
     - ``2``

For example, a reflective pin-cell model with no explicit axial
bounds produces:

.. code-block:: text

   bound_cond 1 1 1 1 1 1

When the original model has no axial boundaries, the two-dimensional
representation uses reflective top and bottom boundaries to avoid
introducing artificial axial leakage.

Boundary inference currently supports axis-aligned rectangular
outer boundaries. More general boundary geometries are not yet
handled by this inference method.

Current Limitations
-------------------

The OpenMC integration is designed primarily for two-dimensional,
Z-invariant geometries.

The real OpenMC converter currently handles the following
surface classes directly:

* ``XPlane``
* ``YPlane``
* ``ZPlane``
* ``ZCylinder``
* ``Plane``

Other surfaces and more complex geometric representations may require
additional implementation.

The separate ``um2.openmc`` package supports additional geometry
features, but this does not imply that all corresponding real OpenMC
models can be converted automatically.

The MPACT hierarchy may use a rectilinear coarse-grid approximation
for models without a directly representable lattice hierarchy.

Geometry outside the represented material regions may be filled
with a synthetic ``Void`` material. The exporter includes a
zero-cross-section dataset for this material.

The resulting files have been validated through UM2 integration
tests, but successful execution and numerical agreement with an
actual MPACT solver remain to be established.

The conversion is not a general-purpose three-dimensional OpenMC
to MPACT translator. In particular, axial modeling requires further
validation beyond the current two-dimensional workflow.

Next Steps
----------

See :doc:`geometry` for the supported geometry features and
limitations.

See :doc:`cross_sections` for further information about multigroup
cross sections.

See :doc:`mpact_export` for details on mesh generation and
MPACT-specific output files.
