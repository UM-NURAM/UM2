# UM2 Python API

An [OpenMC](https://github.com/openmc-dev/openmc)-compatible Python API for
building UM2 models, plus bindings to the UM2 routines that mesh them and write
MPACT models.

- `um2.openmc`: a re-implementation of the parts of the OpenMC Python API that
  describe **multigroup cross sections, materials and 2D (Z-invariant)
  geometry**. OpenMC scripts run largely unchanged.
- `um2.build_geometry`: builds that geometry in gmsh (OpenCASCADE). Each
  material becomes a `Material_<name>` physical group.
- `um2.mpact`: infers the MPACT partition (core, assembly, lattice, RTM,
  coarse cell) from the OpenMC lattices. You can inspect it, edit it, or build
  one by hand.
- `um2.mesh_geometry`: meshes the model with UM2 and writes the MPACT model
  (`.xdmf` plus `.h5`, `.input`, `.volumes` and `.xs`).
- `um2.xs`: converts between OpenMC multigroup libraries (HDF5) and MPACT user
  cross section libraries (`.xsl`).

UM2 and the gmsh Python API work on the **same in-memory gmsh model**, so the
gmsh API can be used at any point, e.g. `gmsh.fltk.run()` to look at the
geometry or mesh.

## Example

```python
import um2
import um2.openmc as openmc

# Materials that reference data sets in a multigroup library
uo2 = openmc.Material(name='uo2')
uo2.add_macroscopic('UO2-3.3')
water = openmc.Material(name='water')
water.add_macroscopic('Moderator')
materials = openmc.Materials([uo2, water])
materials.cross_sections = 'c5g7.xsl'        # or an OpenMC MGXS .h5 file

# A pin cell
fuel_or = openmc.ZCylinder(r=0.54)
box = openmc.model.RectangularPrism(1.26, 1.26, boundary_type='reflective')
pin = openmc.Universe(cells=[openmc.Cell(fill=uo2, region=-fuel_or),
                             openmc.Cell(fill=water, region=+fuel_or & -box)])
model = openmc.Model(geometry=openmc.Geometry(pin), materials=materials)

# Mesh with UM2
plan = um2.mpact.infer_hierarchy(model.geometry)
print(plan)                                   # inspect (and edit) the partition
result = um2.mesh_geometry(model, plan=plan, kn_target=12.0,
                           mesh_type='QuadraticTri', output='pin.xdmf')
```

### Running existing OpenMC scripts

To run a script or notebook that does `import openmc` (and `import openmc.mgxs`
and so on), redirect the import first:

```python
import um2.openmc
um2.openmc.as_openmc()
import openmc                                 # this is now um2.openmc
```

What happens to the rest of an OpenMC script:

- **Ignored:** run-time settings, sources, tallies, plots and `openmc.run()`.
  Using them issues a `UM2IgnoredWarning`.
- **Raise `um2.UM2NotSupportedError`:**
  - features that need OpenMC results, such as `StatePoint`,
    `mgxs.Library.build_library` and depletion;
  - inherently 3D geometry: spheres, cones, x/y-cylinders, rotations about x
    or y, and axial layers.
- **Allowed:** `ZPlane`s that only bound the whole model axially. Their extent
  is recorded, and a vacuum axial boundary produces a warning.

## Supported OpenMC API

| Area | Supported |
|---|---|
| Materials | `Material` (all density units; `add_nuclide`, `add_element` with enrichment, `add_elements_from_formula`, `add_components`, `add_s_alpha_beta` (stored, not used), `add_macroscopic`, `mix_materials`, `get_nuclide_atom_densities`, …), `Materials` (`cross_sections`, XML), `openmc.data` (masses, abundances), `config` |
| Cross sections | `XSdata`, `MGXSLibrary` (OpenMC HDF5 format), `mgxs.EnergyGroups`, `mgxs.GROUP_STRUCTURES`, `Macroscopic` |
| Surfaces | `ZCylinder`, `XPlane`, `YPlane`, `Plane` (with `c = 0`), `ZPlane` (axial bounds only); boundary conditions |
| Composite surfaces | `model.RectangularPrism` (incl. rounded corners), `HexagonalPrism`, `CylinderSector`, `IsogonalOctagon`, `CruciformPrism`, `RectangularParallelepiped`, `model.pin`, `model.subdivide` |
| Regions | `-surf`, `+surf`, `&`, `\|`, `~`, `&=`, `Region.from_expression`, point containment, bounding boxes |
| Cells and universes | `Cell` (material, void, universe and lattice fills; `translation`; `rotation` about z), `Universe`, `Geometry` (queries, `find`, XML export), `Model` |
| Lattices | `RectLattice` (2D, `outer`), `HexLattice` (2D, both orientations, `outer`, `show_indices`) |

The `python/tools/sync_openmc.py check-api` tool (run by the tests) checks
that every class, method and function above has the same signature as in the
OpenMC release pinned by the `extern/openmc` submodule.

## Cross sections

UM2 uses **macroscopic** multigroup cross sections. A material gets them from
the data set named by `Material.add_macroscopic`, looked up in the library
given by `Materials.cross_sections` (or `openmc.config['mg_cross_sections']`,
or the `cross_sections` / `library` arguments of `mesh_geometry`). The library
can be an OpenMC MGXS HDF5 file or an MPACT `.xsl` file.

Materials without macroscopic data (e.g. defined by nuclides) can still be
meshed with a uniform mesh size (`mesh_size=...`). Knudsen-number mesh sizing
(`kn_target=...`) requires cross sections.

Conversion to the MPACT convention (`um2.xs.mpact_xs`, `um2.xs.write_xsl`)
works as follows:

- Scattering matrices are indexed `[to group, from group]`, with P0…PN matrices
  for Legendre data.
- A nu-fission production matrix is summed into a nu-fission vector.
- The diagonal of the P0 scattering matrix is adjusted so that absorption plus
  out-scattering equals the OpenMC total cross section (which is usually
  transport-corrected).
- The lower bound of the lowest energy group is 0 eV.

The `.xsl` format can also be read from C++ with `um2::mpact::UserXSLibrary`.

## The MPACT partition

`um2.mpact.infer_hierarchy(geometry)` builds the partition as follows:

1. **Lattice of lattices** (e.g. C5G7): if the root universe is exactly filled
   by a rectangular lattice of lattices, that lattice is the core and its
   elements are the assemblies.
2. **Lattice of pins:** if it is filled by a lattice of pins, the core is one
   assembly.
3. **Coarse cells:** each lattice element (pin) is a coarse cell and, by
   default, its own RTM (`rtm='pin'`; use `rtm='assembly'` for one RTM per
   assembly).
4. **Large elements:** an element larger than the RTM size, such as a
   reflector assembly defined as a single element, is split into cells of the
   RTM size.
5. **Shared coarse cells:** two positions share a coarse cell, and are meshed
   once, only when they hold the same universe in an element of the same size.
6. **Anything else:** other geometries (hexagonal lattices, lattices surrounded
   by other cells, no lattice) get a rectilinear grid (`um2.mpact.grid_plan`).
   Its lines come from the root cells' x/y planes and lattice boundaries, and
   are subdivided to about the lattice pitch.

The returned `Plan` can be printed and edited, or you can build one by hand
with `add_coarse_cell`, `add_rtm`, `add_lattice`, `add_assembly` and
`set_core`. Arrays of IDs have row 0 at the top, as in OpenMC lattices.

`mesh_geometry` builds only the **unique** coarse cells, so the full 2D C5G7
core meshes in about a second. Space inside the coarse cells but outside the
OpenMC geometry (e.g. beyond a cylindrical or hexagonal boundary) is filled
with a `Void` material that has zero cross sections, unless `fill_material` is
given.

## Building and testing

The C++ extension (`um2._um2`) is built with UM2 when the CMake option
`UM2_BUILD_PYTHON` is on. UM2 must use the **same libgmsh** as the `gmsh`
Python module, so build it in the Python environment you will use it from,
with `GMSH_ROOT` pointing at that environment. With conda:

```bash
conda activate UM2-pythonAPI     # needs gmsh hdf5 pugixml pybind11 numpy h5py pytest
export GMSH_ROOT=$CONDA_PREFIX
mkdir ~/build && cd ~/build
cmake -DUM2_BUILD_PYTHON=ON -DUM2_USE_MPACT_XSLIBS=OFF ~/UM2
make -j
```

(`-DUM2_USE_MPACT_XSLIBS=OFF` skips the C++ tests that need MPACT's
cross-section libraries in `UM2/MPACT_Extras`. Leave it out if you have them.)

Nothing is written to the source tree. The build directory contains a
complete, importable package in `~/build/python/um2`, with the Python sources
copied from `UM2/python/um2` whenever they change (just re-run `make` after
editing them). To use it:

```bash
source ~/build/python/env.sh     # adds ~/build/python to PYTHONPATH
python -c "import um2, um2.openmc"
python -m pytest ~/UM2/python/tests          # or: ctest -R python_tests
```

CMake prints the Python it uses ("UM2 Python bindings: using ...") and warns
if that Python's gmsh module loads a different libgmsh than UM2 links against.
Don't run the tests from inside `UM2/python`: Python would then import the
source copy of `um2`, which has no C++ extension, and most tests would be
skipped.

Alternatively, `pip install ~/UM2/python --no-build-isolation` builds UM2 and
installs the package (with `libum2.so`) into the active environment.

`um2.openmc` and `um2.xs` work without the C++ extension.

The acceptance tests in `tests/test_notebooks.py` run the
[OpenMC example notebooks](https://github.com/openmc-dev/openmc-notebooks)
with `um2.openmc`. They look for the notebooks in `$OPENMC_NOTEBOOKS` or in an
`openmc-notebooks` checkout next to the UM2 repository. Use `-m "not slow"` to
skip the slowest test.

### Updating to a new OpenMC release

```bash
git -C extern/openmc fetch --tags && git -C extern/openmc checkout vX.Y.Z
python python/tools/sync_openmc.py generate     # data tables
python python/tools/sync_openmc.py check-api -v # signature differences
```

Portions of `um2.openmc` are adapted from OpenMC (MIT license; see
`um2/openmc/LICENSE-OpenMC`).
