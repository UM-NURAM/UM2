"""Stubs for parts of the OpenMC API that are ignored by UM2.

UM2 only uses the materials, cross sections and geometry of a model. Run-time
settings, sources, tallies, plots and execution are accepted (so that OpenMC
scripts run unchanged) but have no effect. A :class:`UM2IgnoredWarning` is
issued the first time each ignored feature is used.
"""

from __future__ import annotations

import warnings

from ..errors import UM2NotSupportedError
from . import checkvalue as cv


class UM2IgnoredWarning(UserWarning):
    """A part of an OpenMC script was ignored by UM2."""


_warned = set()


def _ignored(name):
    if name not in _warned:
        _warned.add(name)
        warnings.warn(f'{name} is ignored by UM2, which only uses the materials, '
                      'cross sections and geometry of a model.', UM2IgnoredWarning,
                      stacklevel=3)


def _noop(*args, **kwargs):
    return None


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith('__'):
            raise AttributeError(name)

        # e.g. openmc.Plot.from_geometry(geometry) -> a new stub instance
        def factory(*args, **kwargs):
            return cls()
        return factory


class _Stub(metaclass=_StubMeta):
    """Accepts any constructor arguments, attribute assignments and method
    calls, and ignores them."""

    _um2_stub = True

    def __init__(self, *args, **kwargs):
        _ignored(f'openmc.{type(self).__name__}')
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _noop

    def __repr__(self):
        return f'<{type(self).__name__} (ignored by UM2)>'


class _StubList(cv.CheckedList):
    """A list of stub objects (e.g. Tallies, Plots)."""

    _um2_stub = True
    _item_type = object

    def __init__(self, items=None, *args, **kwargs):
        _ignored(f'openmc.{type(self).__name__}')
        super().__init__(self._item_type, type(self).__name__, items)

    def export_to_xml(self, *args, **kwargs):
        pass


def _make(name, base=_Stub, doc=None):
    return type(name, (base,), {'__doc__': doc or f'Ignored by UM2 (openmc.{name}).'})


# Settings and sources
Settings = _make('Settings')
IndependentSource = _make('IndependentSource')
FileSource = _make('FileSource')
CompiledSource = _make('CompiledSource')
MeshSource = _make('MeshSource')
SourceBase = _make('SourceBase')
VolumeCalculation = _make('VolumeCalculation')
WeightWindows = _make('WeightWindows')
Trigger = _make('Trigger')


def Source(*args, **kwargs):
    """Ignored by UM2 (openmc.Source)."""
    return IndependentSource(*args, **kwargs)


# Meshes
RegularMesh = _make('RegularMesh')
RectilinearMesh = _make('RectilinearMesh')
CylindricalMesh = _make('CylindricalMesh')
SphericalMesh = _make('SphericalMesh')
UnstructuredMesh = _make('UnstructuredMesh')

# Tallies and filters
Tally = _make('Tally')
Filter = _make('Filter')
_FILTERS = ['UniverseFilter', 'MaterialFilter', 'MaterialFromFilter', 'CellFilter',
            'CellFromFilter', 'CellBornFilter', 'CellInstanceFilter', 'CollisionFilter',
            'SurfaceFilter', 'MeshFilter', 'MeshBornFilter', 'MeshSurfaceFilter',
            'EnergyFilter', 'EnergyoutFilter', 'MuFilter', 'MuSurfaceFilter',
            'PolarFilter', 'AzimuthalFilter', 'DistribcellFilter', 'DelayedGroupFilter',
            'EnergyFunctionFilter', 'LegendreFilter', 'SpatialLegendreFilter',
            'SphericalHarmonicsFilter', 'TimeFilter', 'ZernikeFilter',
            'ZernikeRadialFilter', 'ParticleFilter']
for _name in _FILTERS:
    globals()[_name] = _make(_name, Filter)


class Tallies(_StubList):
    """Ignored by UM2 (openmc.Tallies)."""


# Plots
PlotBase = _make('PlotBase')
SlicePlot = _make('SlicePlot', PlotBase)
VoxelPlot = _make('VoxelPlot', PlotBase)
RayTracePlot = _make('RayTracePlot', PlotBase)
WireframeRayTracePlot = _make('WireframeRayTracePlot', RayTracePlot)
SolidRayTracePlot = _make('SolidRayTracePlot', RayTracePlot)
Plot = _make('Plot', PlotBase)


class Plots(_StubList):
    """Ignored by UM2 (openmc.Plots)."""


# Execution
def run(*args, **kwargs):
    """Ignored by UM2 (openmc.run)."""
    _ignored('openmc.run')


def plot_geometry(*args, **kwargs):
    """Ignored by UM2 (openmc.plot_geometry)."""
    _ignored('openmc.plot_geometry')


def plot_inline(*args, **kwargs):
    """Ignored by UM2 (openmc.plot_inline)."""
    _ignored('openmc.plot_inline')


def calculate_volumes(*args, **kwargs):
    """Ignored by UM2 (openmc.calculate_volumes)."""
    _ignored('openmc.calculate_volumes')


def plot_xs(*args, **kwargs):
    """Ignored by UM2 (openmc.plot_xs)."""
    _ignored('openmc.plot_xs')


# Results require running OpenMC
def _requires_openmc(name):
    def __init__(self, *args, **kwargs):
        raise UM2NotSupportedError(f'openmc.{name} requires OpenMC simulation '
                                   'results, which UM2 does not produce.')
    return type(name, (), {'__init__': __init__, '_um2_unsupported': True,
                           '__doc__': f'Not supported by UM2 (openmc.{name}).'})


StatePoint = _requires_openmc('StatePoint')
Summary = _requires_openmc('Summary')
Particle = _requires_openmc('Particle')


def search_for_keff(*args, **kwargs):
    """Not supported by UM2 (requires running OpenMC)."""
    raise UM2NotSupportedError('openmc.search_for_keff requires running OpenMC.')
