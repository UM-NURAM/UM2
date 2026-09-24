"""Mesh an OpenMC-style model with UM2 and export it as an MPACT model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np

from .build import VOID_MATERIAL_NAME, BuildResult, build_geometry
from .colors import material_color
from .mpact import Plan, infer_hierarchy
from .openmc.config import config
from .openmc.geometry import Geometry
from .openmc.material import Materials
from .openmc.model import Model
from .xs import load_library, mpact_xs


def _core():
    from . import _core as core
    return core()


# ==============================================================================
# Materials
# ==============================================================================

def _library_path(materials, cross_sections):
    if cross_sections is not None:
        return Path(cross_sections)
    if materials is not None and materials.cross_sections is not None:
        return Path(materials.cross_sections)
    path = config.get_path('mg_cross_sections')
    return None if path is None else Path(path)


def um2_materials(build: BuildResult, materials=None, cross_sections=None,
                  library=None):
    """Create UM2 materials (with cross sections, if available) for the
    materials of a built geometry.

    Materials that reference a macroscopic data set
    (:meth:`um2.openmc.Material.add_macroscopic`) get their cross sections from
    the multigroup library. The library is ``library`` if given, otherwise it
    is loaded from ``cross_sections``, ``materials.cross_sections`` or
    ``openmc.config['mg_cross_sections']`` (OpenMC HDF5 or MPACT .xsl).

    Returns
    -------
    dict
        Mapping from UM2 material name to um2._um2.Material
    """
    core = _core()
    needs_library = any(m is not None and m.macroscopic is not None
                        for m in build.materials.values())
    if library is None and needs_library:
        path = _library_path(materials, cross_sections)
        if path is not None:
            library = load_library(path)
        else:
            warnings.warn('Materials reference macroscopic data sets, but no '
                          'multigroup library was given (set Materials.cross_sections '
                          "or openmc.config['mg_cross_sections']). The materials "
                          'will not have cross sections.')
    result = {}
    for name, mat in build.materials.items():
        um2_mat = core.Material(name)
        um2_mat.color = material_color(name)
        if mat is not None:
            if mat.temperature is not None:
                um2_mat.temperature = float(mat.temperature)
            if mat.density is not None and mat.density_units in ('g/cm3', 'g/cc'):
                um2_mat.density = float(mat.density)
            if mat.macroscopic is not None and library is not None:
                xsdata = library.get_by_name(mat.macroscopic)
                if xsdata is None:
                    raise KeyError(f'Macroscopic data set "{mat.macroscopic}" of '
                                   f'material "{mat.name}" was not found in the '
                                   f'multigroup library (sets: {library.names})')
                xs = mpact_xs(xsdata, mat.temperature)
                density = mat.density if mat.density_units == 'macro' else 1.0
                if density is None:
                    density = 1.0
                um2_mat.set_xsec(density * xs.absorption, density * xs.fission,
                                 density * xs.nu_fission, density * xs.total,
                                 density * xs.scatter[0])
        result[name] = um2_mat
    _set_void_xsec(result, build)
    return result


def _set_void_xsec(materials, build, void_name=VOID_MATERIAL_NAME):
    """Give void materials (OpenMC cells without a fill) zero cross sections,
    if the other materials have cross sections."""
    groups = {m.num_groups for m in materials.values() if m.has_xsec()}
    if len(groups) > 1:
        raise ValueError(f'Materials have different numbers of groups: {groups}')
    if not groups:
        return
    G = groups.pop()
    zeros = np.zeros(G)
    for name, mat in materials.items():
        is_void = build.materials.get(name, 'not void') is None or name == void_name
        if is_void and not mat.has_xsec():
            mat.set_xsec(zeros, zeros, zeros, zeros, np.zeros((G, G)))


# ==============================================================================
# Meshing
# ==============================================================================

@dataclass
class MeshResult:
    """Result of :func:`mesh_geometry`."""
    #: The UM2 MPACT model (um2._um2.mpact.Model), with meshes and materials
    model: object
    #: The partition that was used
    plan: Plan
    #: The result of building the geometry in gmsh
    build: BuildResult
    #: UM2 materials, by name
    materials: dict = field(default_factory=dict)
    #: Files written
    files: list = field(default_factory=list)


def _mesh_type(mesh_type):
    core = _core()
    if isinstance(mesh_type, str):
        try:
            return getattr(core.MeshType, mesh_type)
        except AttributeError:
            raise ValueError(f'Unknown mesh type "{mesh_type}"; use one of '
                             f'{list(core.MeshType.__members__)}') from None
    return mesh_type


def _check_mesh_quality():
    """Warn if the mesh has inverted elements."""
    import gmsh
    num_inverted = 0
    types, tags, _ = gmsh.model.mesh.getElements(2)
    for elem_tags in tags:
        if len(elem_tags):
            q = gmsh.model.mesh.getElementQualities(elem_tags, 'minSICN')
            num_inverted += int(np.sum(q <= 0))
    if num_inverted:
        warnings.warn(f'The mesh has {num_inverted} inverted element(s). Use a finer '
                      'mesh (larger kn_target or smaller mesh_size) or a triangle '
                      'mesh.')
    return num_inverted


def mesh_geometry(model, *, plan=None, cross_sections=None, library=None,
                  mesh_type='QuadraticTri', mesh_size=None, kn_target=None,
                  fuel_mfp_threshold=-1.0, fuel_mfp_scale=-1.0,
                  abs_mfp_threshold=-1.0, abs_mfp_scale=-1.0,
                  fill_material=None, smooth_iters=100, output=None,
                  write_knudsen_data=False, write_xsec_data=False,
                  mesh_file=None, new_gmsh_model=True, mesh_threads=1):
    """Mesh a model with UM2 and (optionally) write it as an MPACT model.

    The steps are:

    1. Partition the geometry into coarse cells (:func:`um2.mpact.infer_hierarchy`
       unless ``plan`` is given).
    2. Build the geometry of the unique coarse cells in gmsh
       (:func:`um2.build_geometry`).
    3. Create UM2 materials with cross sections from the multigroup library.
    4. Overlay the coarse grid, set the mesh size, and generate the mesh.
    5. Import the coarse cell meshes into the MPACT model and write it.

    The gmsh model remains available afterwards (e.g. ``gmsh.fltk.run()``).

    Parameters
    ----------
    model : um2.openmc.Model or um2.openmc.Geometry
        The model to mesh
    plan : um2.mpact.Plan, optional
        MPACT partition. Inferred if not given.
    cross_sections : path-like, optional
        Multigroup library (OpenMC HDF5 or MPACT .xsl). Defaults to
        ``model.materials.cross_sections`` or
        ``openmc.config['mg_cross_sections']``.
    library : um2.openmc.MGXSLibrary, optional
        Multigroup library, instead of ``cross_sections``
    mesh_type : str or um2.MeshType
        'Tri', 'Quad', 'QuadraticTri' or 'QuadraticQuad'
    mesh_size : float, optional
        Global target mesh size [cm]. Used if ``kn_target`` is not given;
        defaults to 1/10 of the median coarse cell dimension.
    kn_target : float, optional
        Target Knudsen number (mean free path / mean chord length) for mesh
        sizing based on the cross sections. Requires cross sections for all
        materials.
    fuel_mfp_threshold, fuel_mfp_scale, abs_mfp_threshold, abs_mfp_scale : float
        Parameters of the Knudsen number mesh field (see UM2)
    fill_material : str, optional
        Name of the material that fills parts of coarse cells that are not
        covered by the geometry (e.g. outside of a cylindrical or hexagonal
        boundary). Defaults to a void material, ``'Void'``, with zero cross
        sections.
    smooth_iters : int
        Number of mesh smoothing iterations
    output : path-like, optional
        XDMF file to write the MPACT model to (UM2 also writes .h5, .input,
        .volumes and .xs files with the same base name)
    write_knudsen_data, write_xsec_data : bool
        Include Knudsen number / cross section data in the XDMF file
    mesh_file : path-like, optional
        Where to write the intermediate gmsh mesh (Abaqus .inp). A temporary
        file is used by default.
    new_gmsh_model : bool
        Whether to create a new gmsh model before building the geometry
    mesh_threads : int
        Number of threads gmsh uses to generate the mesh (0 = all). The default
        is 1, since gmsh's multithreaded 2D Delaunay meshing intermittently
        fails ("unable to find ...") on models with many small surfaces.

    Returns
    -------
    MeshResult
    """
    import gmsh
    from . import initialize
    core = _core()
    initialize()

    geometry = model.geometry if isinstance(model, Model) else model
    if not isinstance(geometry, Geometry):
        raise TypeError('mesh_geometry expects a um2.openmc.Model or Geometry')
    materials = model.get_all_materials() if isinstance(model, Model) else \
        Materials(geometry.get_all_materials().values())

    if plan is None:
        plan = infer_hierarchy(geometry)
    plan.validate()

    if new_gmsh_model:
        gmsh.model.add(f'um2_{id(plan)}')

    # Build only the unique coarse cells, in MPACT coordinates
    boxes = plan.coarse_cell_boxes()
    clip = [boxes[i] for i in sorted(boxes)]
    build = build_geometry(geometry, clip_boxes=clip, conformal=False,
                           offset=(-plan.origin[0], -plan.origin[1]))
    if not build.materials:
        raise ValueError('The geometry contains no materials')
    covered = sum(build.material_areas().values())
    total = sum((b[2] - b[0]) * (b[3] - b[1]) for b in clip)
    if covered < total * (1 - 1e-6):
        warnings.warn(f'{total - covered:.6g} cm^2 of the coarse cells is not '
                      'covered by the geometry and will be filled with '
                      f'{fill_material or VOID_MATERIAL_NAME!r}')

    um2_mats = um2_materials(build, materials, cross_sections, library)
    if fill_material is None:
        if covered < total * (1 - 1e-6):
            # Space outside of the OpenMC geometry is void
            fill_material = VOID_MATERIAL_NAME
        else:
            # The fill material only fills gaps due to round-off
            areas = build.material_areas()
            fill_material = max(areas, key=areas.get)
    if fill_material == VOID_MATERIAL_NAME and fill_material not in um2_mats:
        void = core.Material(VOID_MATERIAL_NAME)
        void.color = material_color(VOID_MATERIAL_NAME)
        um2_mats[VOID_MATERIAL_NAME] = void
        build.materials[VOID_MATERIAL_NAME] = None
        _set_void_xsec(um2_mats, build)
    if fill_material not in um2_mats:
        raise KeyError(f'Fill material "{fill_material}" is not a material of the '
                       f'geometry ({list(um2_mats)})')

    has_xs = {name: m.has_xsec() for name, m in um2_mats.items()}
    mpact_model = plan.to_um2()
    for name, mat in um2_mats.items():
        mpact_model.add_material(mat, validate=has_xs[name])

    core.gmsh.overlay_coarse_grid(mpact_model, um2_mats[fill_material])

    if kn_target is not None:
        missing = [n for n, h in has_xs.items() if not h]
        if missing:
            raise ValueError(f'kn_target requires cross sections for all materials, '
                             f'but {missing} have none')
        core.gmsh.set_mesh_field_from_knudsen_number(
            2, list(um2_mats.values()), kn_target, fuel_mfp_threshold,
            fuel_mfp_scale, abs_mfp_threshold, abs_mfp_scale)
    else:
        if mesh_size is None:
            mesh_size = float(np.median([min(c.width, c.height)
                                         for c in plan.coarse_cells])) / 10
        core.gmsh.set_global_mesh_size(mesh_size)

    threads = gmsh.option.getNumber('General.NumThreads')
    gmsh.option.setNumber('General.NumThreads', mesh_threads)
    try:
        core.gmsh.generate_mesh(_mesh_type(mesh_type), smooth_iters)
    finally:
        gmsh.option.setNumber('General.NumThreads', threads)
    _check_mesh_quality()

    files = []
    tmpdir = None
    if mesh_file is None:
        tmpdir = tempfile.TemporaryDirectory()
        mesh_file = os.path.join(tmpdir.name, 'um2_mesh.inp')
    else:
        files.append(Path(mesh_file))
    try:
        core.gmsh.write(str(mesh_file))
        mpact_model.import_coarse_cell_meshes(str(mesh_file))
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    if output is not None:
        output = Path(output)
        if output.suffix != '.xdmf':
            raise ValueError('output must be an .xdmf file')
        if (write_knudsen_data or write_xsec_data) and not all(has_xs.values()):
            raise ValueError('Writing Knudsen or cross section data requires cross '
                             'sections for all materials')
        mpact_model.write(str(output), write_knudsen_data, write_xsec_data)
        base = output.with_suffix('')
        files.extend(p for p in (output, base.with_suffix('.h5'),
                                 base.with_suffix('.input'),
                                 base.with_suffix('.volumes'),
                                 base.with_suffix('.xs')) if p.exists())
    return MeshResult(mpact_model, plan, build, um2_mats, files)
