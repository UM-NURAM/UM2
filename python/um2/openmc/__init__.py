"""An OpenMC-compatible Python API for defining UM2 models.

This package re-implements the subset of the `OpenMC <https://openmc.org>`_
Python API needed to describe the materials, multigroup cross sections and
(2D, Z-invariant) geometry of a reactor model, so that OpenMC scripts can be
used with UM2 largely unchanged::

    import um2.openmc as openmc

    fuel = openmc.Material(name='fuel')
    fuel.add_macroscopic('uo2')
    ...

To run scripts that ``import openmc`` (including ``import openmc.mgxs`` etc.)
unchanged, call :func:`as_openmc` first::

    import um2.openmc
    um2.openmc.as_openmc()
    import openmc          # this is now um2.openmc

Run-time features (settings, sources, tallies, plots, running OpenMC) are
accepted but ignored, and inherently 3D features (e.g. spheres) raise
:class:`um2.errors.UM2NotSupportedError`.

Portions of this package are adapted from OpenMC, which is distributed under
the MIT license (see LICENSE-OpenMC). The data tables are generated from the
OpenMC release recorded in :data:`OPENMC_VERSION` by
``python/tools/sync_openmc.py``.
"""

import sys

from ..errors import UM2NotSupportedError
from . import checkvalue, data, deplete, mgxs, model, stats
from ._openmc_version import OPENMC_VERSION
from ._stubs import (CompiledSource, FileSource, Filter, IndependentSource,
                     MeshSource, Particle, Plot, PlotBase, Plots, RayTracePlot,
                     RectilinearMesh, RegularMesh, Settings, SlicePlot,
                     SolidRayTracePlot, Source, SourceBase, StatePoint, Summary,
                     Tallies, Tally, Trigger, UM2IgnoredWarning, UnstructuredMesh,
                     VolumeCalculation, VoxelPlot, WeightWindows,
                     WireframeRayTracePlot, CylindricalMesh, SphericalMesh,
                     calculate_volumes, plot_geometry, plot_inline, plot_xs, run,
                     search_for_keff)
from ._stubs import _FILTERS
from . import _stubs as _stubs_module
from .bounding_box import BoundingBox
from .cell import Cell
from .config import config
from .element import Element
from .geometry import Geometry
from .lattice import HexLattice, Lattice, RectLattice
from .macroscopic import Macroscopic
from .material import Material, Materials
from .mgxs_library import MGXSLibrary, XSdata
from .mixin import IDWarning, reserve_ids, reset_auto_ids, set_auto_id
from .model import Model
from .region import Complement, Intersection, Region, Union
from .surface import (Cone, Cylinder, Halfspace, Plane, Quadric, Sphere, Surface,
                      XCone, XCylinder, XPlane, XTorus, YCone, YCylinder, YPlane,
                      YTorus, ZCone, ZCylinder, ZPlane, ZTorus)
from .universe import DAGMCUniverse, Universe, UniverseBase

for _name in _FILTERS:
    globals()[_name] = getattr(_stubs_module, _name)

#: Version of the OpenMC API that this package mirrors
__version__ = OPENMC_VERSION.lstrip('v').split('-')[0]

_SUBMODULES = ('checkvalue', 'data', 'deplete', 'mgxs', 'model', 'stats')


def as_openmc():
    """Make ``import openmc`` (and its submodules) import this package.

    Returns
    -------
    module
        This module (``um2.openmc``)
    """
    this = sys.modules[__name__]
    existing = sys.modules.get('openmc')
    if existing is not None and existing is not this:
        raise RuntimeError('The real openmc package has already been imported; '
                           'call um2.openmc.as_openmc() before importing openmc.')
    sys.modules['openmc'] = this
    for sub in _SUBMODULES:
        sys.modules[f'openmc.{sub}'] = getattr(this, sub)
    sys.modules['openmc.mgxs.groups'] = mgxs.groups
    sys.modules['openmc.model.surface_composite'] = model.surface_composite
    sys.modules['openmc.model.funcs'] = model.funcs
    return this
