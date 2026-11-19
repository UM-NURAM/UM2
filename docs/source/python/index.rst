==========
Python API
==========

UM2 provides a Python interface for defining reactor geometries,
generating unstructured meshes, handling multigroup cross sections,
and exporting MPACT-compatible models.

Two workflows are supported:

* **UM2's OpenMC-compatible API:** Construct models using
  ``um2.openmc`` and generate meshes with ``um2.mesh_geometry``.
* **Real OpenMC integration:** Construct models using the actual
  ``openmc`` package and convert supported models with
  ``um2.generate_um2_from_openmc``.

The Python API is primarily designed for two-dimensional,
Z-invariant neutron transport geometries.

.. toctree::
   :maxdepth: 3
   :caption: Python API Documentation

   installation
   quickstart
   openmc_integration
   geometry
   cross_sections
   mpact_export
   api_reference
   development
