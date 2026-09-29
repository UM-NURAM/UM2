"""Stub of :mod:`openmc.stats` (source distributions are ignored by UM2)."""

from __future__ import annotations

from ._stubs import _Stub, _make

_NAMES = ['Univariate', 'Discrete', 'Uniform', 'PowerLaw', 'Maxwell', 'Watt',
          'Tabular', 'Legendre', 'Mixture', 'Normal', 'Muir', 'delta_function',
          'UnitSphere', 'PolarAzimuthal', 'Isotropic', 'Monodirectional',
          'Spatial', 'CartesianIndependent', 'SphericalIndependent',
          'CylindricalIndependent', 'MeshSpatial', 'PointCloud', 'Box', 'Point']
for _name in _NAMES:
    globals()[_name] = _make(_name, _Stub, f'Ignored by UM2 (openmc.stats.{_name}).')

__all__ = list(_NAMES)
