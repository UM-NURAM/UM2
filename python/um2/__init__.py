"""UM2: University of Michigan Unstructured Mesh code.

The :mod:`um2.openmc` package provides an OpenMC-compatible API for defining
materials, multigroup cross sections and 2D geometry. The functions in this
module build that geometry in gmsh, partition it into an MPACT model, mesh it
and export it with UM2::

    import um2
    import um2.openmc as openmc
    ...                                   # an OpenMC model
    plan = um2.mpact.infer_hierarchy(model.geometry)
    print(plan)                           # inspect / edit the MPACT partition
    result = um2.mesh_geometry(model, plan=plan, kn_target=12.0,
                               output='model.xdmf')

UM2 and the gmsh Python API share the same gmsh model, so the gmsh API can be
used at any point (e.g. ``gmsh.fltk.run()`` to view the model).
"""

from .errors import UM2Error, UM2NotSupportedError
from . import mpact, xs
from .build import build_geometry, material_areas, material_names
from .colors import material_color, set_material_color
from .pipeline import MeshResult, mesh_geometry, um2_materials

_initialized = False


def _core():
    """Return the compiled UM2 extension module (um2._um2)."""
    try:
        from . import _um2
    except ImportError as e:
        raise ImportError(
            'The UM2 C++ extension (um2._um2) is not available. Build it with '
            '"pip install ./python" or by configuring UM2 with '
            '-DUM2_BUILD_PYTHON=ON. Only um2.openmc and um2.xs can be used '
            'without it.') from e
    return _um2


def initialize(log_level=None):
    """Initialize UM2 and gmsh (if they are not already initialized).

    Parameters
    ----------
    log_level : int, optional
        UM2 log level: 0 = off, 1 = error, 2 = warn, 3 = info, 4 = debug
    """
    global _initialized
    core = _core()
    if not _initialized:
        core.initialize()
        _initialized = True
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    if log_level is not None:
        core.set_log_level(log_level)


def finalize(finalize_gmsh=True):
    """Finalize UM2 and, by default, gmsh."""
    global _initialized
    import gmsh
    if _initialized:
        _core().finalize(False)
        _initialized = False
    if finalize_gmsh and gmsh.isInitialized():
        gmsh.finalize()


def __getattr__(name):
    # Expose MeshType and Material from the extension lazily, so that
    # um2.openmc can be used without the compiled extension.
    if name in ('MeshType', 'Material', 'UserXSLibrary'):
        return getattr(_core(), name)
    raise AttributeError(f"module 'um2' has no attribute '{name}'")


__all__ = ['UM2Error', 'UM2NotSupportedError', 'initialize', 'finalize',
           'build_geometry', 'material_areas', 'material_names', 'mesh_geometry', 'MeshResult',
           'um2_materials', 'material_color', 'set_material_color', 'mpact', 'xs',
           'MeshType', 'Material', 'UserXSLibrary']
