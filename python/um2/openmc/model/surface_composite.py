"""Composite surfaces (subset of :mod:`openmc.model.surface_composite`).

Only composite surfaces that are Z-invariant (``axis='z'``) are supported.
:class:`RectangularParallelepiped` is supported, but its z-planes may only be
used to bound the model axially.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import copy
from functools import partial
from math import cos, pi, sin, sqrt
from numbers import Real

import numpy as np

from ...errors import UM2NotSupportedError
from ..checkvalue import check_type, check_value
from ..region import Union
from ..surface import Plane, XPlane, YPlane, ZCylinder, ZPlane


def _check_z_axis(name, axis):
    if axis != 'z':
        raise UM2NotSupportedError(
            f"{name}(axis='{axis}') is not supported by UM2, since the geometry "
            "must be Z-invariant. Use axis='z'.")


class CompositeSurface(ABC):
    """Multiple primitive surfaces combined into a composite surface"""


    def translate(self, vector, inplace=False):
        surf = self if inplace else copy(self)
        for name in self._surface_names:
            s = getattr(surf, name)
            setattr(surf, name, s.translate(vector, inplace))
        return surf

    def rotate(self, rotation, pivot=(0.0, 0.0, 0.0), order='xyz', inplace=False):
        surf = self if inplace else copy(self)
        for name in self._surface_names:
            s = getattr(surf, name)
            setattr(surf, name, s.rotate(rotation, pivot, order, inplace))
        return surf

    @property
    def component_surfaces(self):
        return [getattr(self, name) for name in self._surface_names]

    @property
    def boundary_type(self):
        return getattr(self, self._surface_names[0]).boundary_type

    @boundary_type.setter
    def boundary_type(self, boundary_type):
        for name in self._surface_names:
            getattr(self, name).boundary_type = boundary_type

    def __repr__(self):
        return f'<{type(self).__name__} at 0x{id(self):x}>'

    @property
    @abstractmethod
    def _surface_names(self):
        pass

    @abstractmethod
    def __neg__(self) -> Region:
        pass

    def __pos__(self) -> Region:
        return ~-self

class CylinderSector(CompositeSurface):
    """Infinite cylindrical sector composite surface (axis must be z)."""

    _surface_names = ('outer_cyl', 'inner_cyl', 'plane1', 'plane2')

    def __init__(self, r1, r2, theta1, theta2, center=(0.0, 0.0), axis='z', **kwargs):
        if r2 <= r1:
            raise ValueError('r2 must be greater than r1.')
        if theta2 <= theta1:
            raise ValueError('theta2 must be greater than theta1.')
        _check_z_axis(type(self).__name__, axis)
        self._reflex = theta2 - theta1 > 180.0
        phi1 = pi / 180 * theta1
        phi2 = pi / 180 * theta2
        p1 = np.array([center[0], center[1], 1.0])
        p2_plane1 = np.array([r1 * cos(phi1) + center[0], r1 * sin(phi1) + center[1], 0.0])
        p3_plane1 = np.array([r2 * cos(phi1) + center[0], r2 * sin(phi1) + center[1], 0.0])
        p2_plane2 = np.array([r1 * cos(phi2) + center[0], r1 * sin(phi2) + center[1], 0.0])
        p3_plane2 = np.array([r2 * cos(phi2) + center[0], r2 * sin(phi2) + center[1], 0.0])
        points = [p1, p2_plane1, p3_plane1, p2_plane2, p3_plane2]
        self.inner_cyl = ZCylinder(*center, r1, **kwargs)
        self.outer_cyl = ZCylinder(*center, r2, **kwargs)
        self.plane1 = Plane.from_points(p1, p2_plane1, p3_plane1, **kwargs)
        self.plane2 = Plane.from_points(p1, p2_plane2, p3_plane2, **kwargs)

    @classmethod
    def from_theta_alpha(cls, r1, r2, theta, alpha, center=(0.0, 0.0), axis='z', **kwargs):
        if theta >= 360.0 or theta <= 0:
            raise ValueError('theta must be less than 360 and greater than 0.')
        theta1 = alpha
        theta2 = alpha + theta
        return cls(r1, r2, theta1, theta2, center=center, axis=axis, **kwargs)

    def __neg__(self):
        if self._reflex:
            return -self.outer_cyl & +self.inner_cyl & (-self.plane1 | +self.plane2)
        else:
            return -self.outer_cyl & +self.inner_cyl & -self.plane1 & +self.plane2

class IsogonalOctagon(CompositeSurface):
    """Infinite isogonal octagon composite surface (axis must be z)."""

    _surface_names = ('top', 'bottom', 'upper_right', 'lower_left', 'right', 'left', 'lower_right', 'upper_left')

    def __init__(self, center, r1, r2, axis='z', **kwargs):
        _check_z_axis(type(self).__name__, axis)
        c1, c2 = center
        cright = c1 + r1
        cleft = c1 - r1
        ctop = c2 + r1
        cbottom = c2 - r1
        if r2 > r1 * sqrt(2):
            raise ValueError('r2 is greater than sqrt(2) * r1. Octagon' + ' may be erroneous.')
        if r1 > r2 * sqrt(2):
            raise ValueError('r1 is greater than sqrt(2) * r2. Octagon' + ' may be erroneous.')
        L_basis_ax = r2 * sqrt(2) - r1
        p1_ur = np.array([L_basis_ax, r1, 0.0])
        p2_ur = np.array([r1, L_basis_ax, 0.0])
        p3_ur = np.array([r1, L_basis_ax, 1.0])
        p1_lr = np.array([r1, -L_basis_ax, 0.0])
        p2_lr = np.array([L_basis_ax, -r1, 0.0])
        p3_lr = np.array([L_basis_ax, -r1, 1.0])
        p1_ll = -p1_ur
        p2_ll = -p2_ur
        p3_ll = -p3_ur
        p1_ul = -p1_lr
        p2_ul = -p2_lr
        p3_ul = -p3_lr
        points = [p1_ur, p2_ur, p3_ur, p1_lr, p2_lr, p3_lr, p1_ll, p2_ll, p3_ll, p1_ul, p2_ul, p3_ul]
        self.top = YPlane(ctop, **kwargs)
        self.bottom = YPlane(cbottom, **kwargs)
        self.right = XPlane(cright, **kwargs)
        self.left = XPlane(cleft, **kwargs)
        self.axis = axis
        for p in points:
            p[0] += c1
            p[1] += c2
        self.upper_right = Plane.from_points(p1_ur, p2_ur, p3_ur, **kwargs)
        self.lower_right = Plane.from_points(p1_lr, p2_lr, p3_lr, **kwargs)
        self.lower_left = Plane.from_points(p1_ll, p2_ll, p3_ll, **kwargs)
        self.upper_left = Plane.from_points(p1_ul, p2_ul, p3_ul, **kwargs)

    def __neg__(self):
        if self.axis == 'y':
            region = -self.top & +self.bottom & -self.right & +self.left & -self.upper_right & -self.lower_right & +self.lower_left & +self.upper_left
        else:
            region = -self.top & +self.bottom & -self.right & +self.left & +self.upper_right & +self.lower_right & -self.lower_left & -self.upper_left
        return region

    def __pos__(self):
        if self.axis == 'y':
            region = +self.top | -self.bottom | +self.right | -self.left | +self.upper_right | +self.lower_right | -self.lower_left | -self.upper_left
        else:
            region = +self.top | -self.bottom | +self.right | -self.left | -self.upper_right | -self.lower_right | +self.lower_left | +self.upper_left
        return region

class RectangularParallelepiped(CompositeSurface):
    """Rectangular parallelpiped composite surface. The z-planes may only bound the model axially."""

    _surface_names = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, **kwargs):
        if xmin >= xmax:
            raise ValueError('xmin must be less than xmax')
        if ymin >= ymax:
            raise ValueError('ymin must be less than ymax')
        if zmin >= zmax:
            raise ValueError('zmin must be less than zmax')
        self.xmin = XPlane(x0=xmin, **kwargs)
        self.xmax = XPlane(x0=xmax, **kwargs)
        self.ymin = YPlane(y0=ymin, **kwargs)
        self.ymax = YPlane(y0=ymax, **kwargs)
        self.zmin = ZPlane(z0=zmin, **kwargs)
        self.zmax = ZPlane(z0=zmax, **kwargs)

    def __neg__(self):
        return -self.xmax & +self.xmin & -self.ymax & +self.ymin & -self.zmax & +self.zmin

    def __pos__(self):
        return +self.xmax | -self.xmin | +self.ymax | -self.ymin | +self.zmax | -self.zmin

class CruciformPrism(CompositeSurface):
    """Generalized cruciform prism composite surface (axis must be z)."""


    def __init__(self, distances, center=(0.0, 0.0), axis='z', **kwargs):
        _check_z_axis(type(self).__name__, axis)
        x0, y0 = center
        self.distances = distances
        cls_horizontal = XPlane
        cls_vertical = YPlane
        surfnames = []
        for i, d in enumerate(distances):
            setattr(self, f'hmin{i}', cls_horizontal(x0 - d, **kwargs))
            setattr(self, f'hmax{i}', cls_horizontal(x0 + d, **kwargs))
            setattr(self, f'vmin{i}', cls_vertical(y0 - d, **kwargs))
            setattr(self, f'vmax{i}', cls_vertical(y0 + d, **kwargs))
            surfnames.extend([f'hmin{i}', f'hmax{i}', f'vmin{i}', f'vmax{i}'])
        self._surfnames = tuple(surfnames)

    @property
    def _surface_names(self):
        return self._surfnames

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, values):
        values = np.array(values, dtype=float)
        if not (values > 0).all():
            raise ValueError('distances must be positive')
        if (values[1:] > values[:-1]).all() or (values[1:] < values[:-1]).all():
            self._distances = values
        else:
            raise ValueError('distances must be monotonic')

    def __neg__(self):
        n = len(self.distances)
        regions = []
        for i in range(n):
            regions.append(+getattr(self, f'hmin{i}') & -getattr(self, f'hmax{i}') & +getattr(self, f'vmin{n - 1 - i}') & -getattr(self, f'vmax{n - 1 - i}'))
        return Union(regions)

def _plane(axis, name, value, boundary_type='transmission', albedo=1.0):
    """Create an axis-aligned plane"""
    cls = {'x': XPlane, 'y': YPlane, 'z': ZPlane}[axis]
    return cls(value, name=f'{name} {axis}', boundary_type=boundary_type, albedo=albedo)

class RectangularPrism(CompositeSurface):
    """Infinite rectangular prism bounded by four planar surfaces (axis must be z)."""

    _surface_names = ('min_x1', 'max_x1', 'min_x2', 'max_x2')

    def __init__(self, width: float, height: float, axis: str='z', origin: Sequence[float]=(0.0, 0.0), boundary_type: str='transmission', albedo: float=1.0, corner_radius: float=0.0):
        check_type('width', width, Real)
        check_type('height', height, Real)
        check_type('albedo', albedo, Real)
        check_type('corner_radius', corner_radius, Real)
        check_value('axis', axis, ('x', 'y', 'z'))
        _check_z_axis(type(self).__name__, axis)
        check_type('origin', origin, Iterable, Real)
        if axis == 'x':
            x1, x2 = ('y', 'z')
        elif axis == 'y':
            x1, x2 = ('x', 'z')
        else:
            x1, x2 = ('x', 'y')
        cyl = ZCylinder
        bc_args = {'boundary_type': boundary_type, 'albedo': albedo}
        self.min_x1 = _plane(x1, 'minimum', -width / 2 + origin[0], **bc_args)
        self.max_x1 = _plane(x1, 'maximum', width / 2 + origin[0], **bc_args)
        self.min_x2 = _plane(x2, 'minimum', -height / 2 + origin[1], **bc_args)
        self.max_x2 = _plane(x2, 'maximum', height / 2 + origin[1], **bc_args)
        if boundary_type == 'periodic':
            self.min_x1.periodic_surface = self.max_x1
            self.min_x2.periodic_surface = self.max_x2
        if corner_radius > 0.0:
            if boundary_type == 'periodic':
                raise ValueError('Periodic boundary conditions not permitted when rounded corners are used.')
            args = {'r': corner_radius, 'boundary_type': boundary_type, 'albedo': albedo}
            args[x1 + '0'] = origin[0] - width / 2 + corner_radius
            args[x2 + '0'] = origin[1] - height / 2 + corner_radius
            self.x1_min_x2_min = cyl(name=f'{x1} min {x2} min', **args)
            args[x1 + '0'] = origin[0] - width / 2 + corner_radius
            args[x2 + '0'] = origin[1] + height / 2 - corner_radius
            self.x1_min_x2_max = cyl(name=f'{x1} min {x2} max', **args)
            args[x1 + '0'] = origin[0] + width / 2 - corner_radius
            args[x2 + '0'] = origin[1] - height / 2 + corner_radius
            self.x1_max_x2_min = cyl(name=f'{x1} max {x2} min', **args)
            args[x1 + '0'] = origin[0] + width / 2 - corner_radius
            args[x2 + '0'] = origin[1] + height / 2 - corner_radius
            self.x1_max_x2_max = cyl(name=f'{x1} max {x2} max', **args)
            self.x1_min = _plane(x1, 'min', -width / 2 + origin[0] + corner_radius, **bc_args)
            self.x1_max = _plane(x1, 'max', width / 2 + origin[0] - corner_radius, **bc_args)
            self.x2_min = _plane(x2, 'min', -height / 2 + origin[1] + corner_radius, **bc_args)
            self.x2_max = _plane(x2, 'max', height / 2 + origin[1] - corner_radius, **bc_args)
            self._surface_names += ('x1_min_x2_min', 'x1_min_x2_max', 'x1_max_x2_min', 'x1_max_x2_max', 'x1_min', 'x1_max', 'x2_min', 'x2_max')

    def __neg__(self):
        prism = +self.min_x1 & -self.max_x1 & +self.min_x2 & -self.max_x2
        if hasattr(self, 'x1_min'):
            corners = +self.x1_min_x2_min & -self.x1_min & -self.x2_min | +self.x1_min_x2_max & -self.x1_min & +self.x2_max | +self.x1_max_x2_min & +self.x1_max & -self.x2_min | +self.x1_max_x2_max & +self.x1_max & +self.x2_max
            prism &= ~corners
        return prism

class HexagonalPrism(CompositeSurface):
    """Hexagonal prism comoposed of six planar surfaces."""

    _surface_names = ('plane_max', 'plane_min', 'upper_right', 'upper_left', 'lower_right', 'lower_left')

    def __init__(self, edge_length: float=1.0, orientation: str='y', origin: Sequence[float]=(0.0, 0.0), boundary_type: str='transmission', albedo: float=1.0, corner_radius: float=0.0):
        check_type('edge_length', edge_length, Real)
        check_type('albedo', albedo, Real)
        check_type('corner_radius', corner_radius, Real)
        check_value('orientation', orientation, ('x', 'y'))
        check_type('origin', origin, Iterable, Real)
        l = edge_length
        x, y = origin
        bc_args = {'boundary_type': boundary_type, 'albedo': albedo}
        if orientation == 'y':
            self.plane_max = XPlane(x + sqrt(3.0) / 2 * l, **bc_args)
            self.plane_min = XPlane(x - sqrt(3.0) / 2 * l, **bc_args)
            c = sqrt(3.0) / 3.0
            self.upper_right = Plane(a=c, b=1.0, d=l + x * c + y, **bc_args)
            self.upper_left = Plane(a=-c, b=1.0, d=l - x * c + y, **bc_args)
            self.lower_right = Plane(a=-c, b=1.0, d=-l - x * c + y, **bc_args)
            self.lower_left = Plane(a=c, b=1.0, d=-l + x * c + y, **bc_args)
        elif orientation == 'x':
            self.plane_max = YPlane(y + sqrt(3.0) / 2 * l, **bc_args)
            self.plane_min = YPlane(y - sqrt(3.0) / 2 * l, **bc_args)
            c = sqrt(3.0)
            self.upper_right = Plane(a=c, b=1.0, d=c * l + x * c + y, **bc_args)
            self.lower_right = Plane(a=-c, b=1.0, d=-c * l - x * c + y, **bc_args)
            self.lower_left = Plane(a=c, b=1.0, d=-c * l + x * c + y, **bc_args)
            self.upper_left = Plane(a=-c, b=1.0, d=c * l - x * c + y, **bc_args)
        if boundary_type == 'periodic':
            self.plane_min.periodic_surface = self.plane_max
            self.upper_right.periodic_surface = self.lower_left
            self.lower_right.periodic_surface = self.upper_left
        if corner_radius > 0.0:
            if boundary_type == 'periodic':
                raise ValueError('Periodic boundary conditions not permitted when rounded corners are used.')
            c = sqrt(3.0) / 2
            t = l - corner_radius / c
            cyl1 = partial(ZCylinder, r=corner_radius, **bc_args)
            cyl2 = partial(ZCylinder, r=corner_radius / (2 * c), **bc_args)
            if orientation == 'x':
                self.x_min_y_min_in = cyl1(name='x min y min in', x0=x - t / 2, y0=y - c * t)
                self.x_min_y_max_in = cyl1(name='x min y max in', x0=x + t / 2, y0=y - c * t)
                self.x_max_y_min_in = cyl1(name='x max y min in', x0=x - t / 2, y0=y + c * t)
                self.x_max_y_max_in = cyl1(name='x max y max in', x0=x + t / 2, y0=y + c * t)
                self.min_in = cyl1(name='x min in', x0=x - t, y0=y)
                self.max_in = cyl1(name='x max in', x0=x + t, y0=y)
                self.x_min_y_min_out = cyl2(name='x min y min out', x0=x - l / 2, y0=y - c * l)
                self.x_min_y_max_out = cyl2(name='x min y max out', x0=x + l / 2, y0=y - c * l)
                self.x_max_y_min_out = cyl2(name='x max y min out', x0=x - l / 2, y0=y + c * l)
                self.x_max_y_max_out = cyl2(name='x max y max out', x0=x + l / 2, y0=y + c * l)
                self.min_out = cyl2(name='x min out', x0=x - l, y0=y)
                self.max_out = cyl2(name='x max out', x0=x + l, y0=y)
            elif orientation == 'y':
                self.x_min_y_min_in = cyl1(name='x min y min in', x0=x - c * t, y0=y - t / 2)
                self.x_min_y_max_in = cyl1(name='x min y max in', x0=x - c * t, y0=y + t / 2)
                self.x_max_y_min_in = cyl1(name='x max y min in', x0=x + c * t, y0=y - t / 2)
                self.x_max_y_max_in = cyl1(name='x max y max in', x0=x + c * t, y0=y + t / 2)
                self.min_in = cyl1(name='y min in', x0=x, y0=y - t)
                self.max_in = cyl1(name='y max in', x0=x, y0=y + t)
                self.x_min_y_min_out = cyl2(name='x min y min out', x0=x - c * l, y0=y - l / 2)
                self.x_min_y_max_out = cyl2(name='x min y max out', x0=x - c * l, y0=y + l / 2)
                self.x_max_y_min_out = cyl2(name='x max y min out', x0=x + c * l, y0=y - l / 2)
                self.x_max_y_max_out = cyl2(name='x max y max out', x0=x + c * l, y0=y + l / 2)
                self.min_out = cyl2(name='y min out', x0=x, y0=y - l)
                self.max_out = cyl2(name='y max out', x0=x, y0=y + l)
            for s in ('in', 'out'):
                self._surface_names += (f'x_min_y_min_{s}', f'x_min_y_max_{s}', f'x_max_y_min_{s}', f'x_max_y_max_{s}', f'min_{s}', f'max_{s}')

    def __neg__(self) -> Region:
        prism = -self.plane_max & +self.plane_min & -self.upper_right & -self.upper_left & +self.lower_right & +self.lower_left
        if hasattr(self, 'min_in'):
            corners = +self.x_min_y_min_in & -self.x_min_y_min_out | +self.x_min_y_max_in & -self.x_min_y_max_out | +self.x_max_y_min_in & -self.x_max_y_min_out | +self.x_max_y_max_in & -self.x_max_y_max_out | +self.min_in & -self.min_out | +self.max_in & -self.max_out
            prism &= ~corners
        return prism


def _unsupported(name, reason):
    def __init__(self, *args, **kwargs):
        raise UM2NotSupportedError(f'openmc.model.{name} is not supported by UM2: '
                                   f'{reason}.')
    return type(name, (), {'__init__': __init__, '_um2_unsupported': True,
                           '__doc__': f'Not supported by UM2 ({reason}).'})


XConeOneSided = _unsupported('XConeOneSided', 'cones are not Z-invariant')
YConeOneSided = _unsupported('YConeOneSided', 'cones are not Z-invariant')
ZConeOneSided = _unsupported('ZConeOneSided', 'cones are not Z-invariant')
ConicalFrustum = _unsupported('ConicalFrustum', 'cones are not Z-invariant')
RightCircularCylinder = _unsupported('RightCircularCylinder',
                                     'finite cylinders are not Z-invariant')
Vessel = _unsupported('Vessel', 'vessels with ellipsoidal heads are not Z-invariant')
OrthogonalBox = _unsupported('OrthogonalBox', 'use RectangularPrism instead')
Polygon = _unsupported('Polygon', 'polygons are not supported yet')
