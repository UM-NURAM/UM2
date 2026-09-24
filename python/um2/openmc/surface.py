"""Surfaces (compatible with :mod:`openmc.surface`).

UM2 generates 2D meshes of Z-invariant geometry, so only surfaces that are
parallel to the z-axis can be used to define the radial geometry:
:class:`Plane` (with ``c == 0``), :class:`XPlane`, :class:`YPlane` and
:class:`ZCylinder`. :class:`ZPlane` (and ``Plane`` with ``a == b == 0``) may be
used only to bound the whole model axially; the axial extent is recorded but
otherwise ignored. All other OpenMC surfaces raise
:class:`um2.errors.UM2NotSupportedError` when they are created.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
import math
from numbers import Real
from warnings import warn
import xml.etree.ElementTree as ET

import numpy as np

from ..errors import UM2NotSupportedError
from ._xml import get_elem_list, get_text
from .bounding_box import BoundingBox
from .checkvalue import check_greater_than, check_length, check_type, check_value
from .mixin import IDManagerMixin
from .region import Intersection, Region, Union

_BOUNDARY_TYPES = {'transmission', 'vacuum', 'reflective', 'periodic', 'white'}
_ALBEDO_BOUNDARIES = {'reflective', 'periodic', 'white'}

_WARNING_UPPER = """\
"{}(...) accepts an argument named '{}', not '{}'. Future versions of OpenMC \
will not accept the capitalized version.\
"""

_WARNING_KWARGS = """\
"{}(...) accepts keyword arguments only for '{}'. Future versions of OpenMC \
will not accept positional parameters for superclass arguments.\
"""


class SurfaceCoefficient:
    """Descriptor class for surface coefficients.

    Parameters
    -----------
    value : float or str
        Value of the coefficient (float) or the name of the coefficient that
        it is equivalent to (str).
    positive : bool
        Whether the coefficient must be positive
    """

    def __init__(self, value, positive=False):
        self.value = value
        self.positive = positive

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if isinstance(self.value, str):
            return instance._coefficients[self.value]
        return self.value

    def __set__(self, instance, value):
        if isinstance(self.value, Real):
            raise AttributeError('This coefficient is read-only')
        check_type(f'{self.value} coefficient', value, Real)
        if self.positive:
            check_greater_than(f'{self.value} coefficient', value, 0.0)
        instance._coefficients[self.value] = value


def _future_kwargs_warning_helper(cls, *args, **kwargs):
    # Warn if Surface parameters are passed by position, not by keyword
    argsdict = dict(zip(('boundary_type', 'name', 'surface_id'), args))
    for k in argsdict:
        warn(_WARNING_KWARGS.format(cls.__name__, k), FutureWarning)
    kwargs.update(argsdict)
    return kwargs


def get_rotation_matrix(rotation, order='xyz'):
    r"""Generate a 3x3 rotation matrix from input angles (in degrees)."""
    check_type('surface rotation', rotation, Iterable, Real)
    check_length('surface rotation', rotation, 3)
    phi, theta, psi = np.array(rotation)*(math.pi/180.)
    cx, sx = math.cos(phi), math.sin(phi)
    cy, sy = math.cos(theta), math.sin(theta)
    cz, sz = math.cos(psi), math.sin(psi)
    R = {
        'x': np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]]),
        'y': np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]]),
        'z': np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]]),
    }
    R1, R2, R3 = (R[xi] for xi in order)
    return R3 @ R2 @ R1


def _rotation_matrix(rotation, order):
    rotation = np.asarray(rotation, dtype=float)
    if rotation.ndim == 2:
        check_length('surface rotation', rotation.ravel(), 9)
        return rotation
    return get_rotation_matrix(rotation, order=order)


def _check_z_rotation(Rmat, what):
    """Ensure that a rotation matrix only rotates about the z-axis."""
    if not (np.allclose(Rmat[2], (0., 0., 1.), atol=1e-12)
            and np.allclose(Rmat[:, 2], (0., 0., 1.), atol=1e-12)):
        raise UM2NotSupportedError(
            f'{what}: UM2 only supports rotations about the z-axis, since the '
            'geometry must be Z-invariant.')


class Surface(IDManagerMixin, ABC):
    """An implicit surface with an associated boundary condition.

    Parameters
    ----------
    surface_id : int, optional
        Unique identifier for the surface. If not specified, an identifier will
        automatically be assigned.
    boundary_type : {'transmission', 'vacuum', 'reflective', 'periodic', 'white'}, optional
        Boundary condition that defines the behavior for particles hitting the
        surface. Defaults to transmissive boundary condition where particles
        freely pass through the surface.
    albedo : float, optional
        Albedo of the surfaces as a ratio of particle weight after interaction
        with the surface to the initial weight.
    name : str, optional
        Name of the surface. If not specified, the name will be the empty
        string.
    """

    next_id = 1
    used_ids = set()
    _atol = 1.e-12

    def __init__(self, surface_id=None, boundary_type='transmission',
                 albedo=1., name=''):
        self.id = surface_id
        self.name = name
        self.boundary_type = boundary_type
        self.albedo = albedo
        self._coefficients = {}

    def __neg__(self):
        return Halfspace(self, '-')

    def __pos__(self):
        return Halfspace(self, '+')

    def __repr__(self):
        string = 'Surface\n'
        string += '{0: <20}{1}{2}\n'.format('\tID', '=\t', self._id)
        string += '{0: <20}{1}{2}\n'.format('\tName', '=\t', self._name)
        string += '{0: <20}{1}{2}\n'.format('\tType', '=\t', self._type)
        string += '{0: <20}{1}{2}\n'.format('\tBoundary', '=\t', self._boundary_type)
        if (self._boundary_type in _ALBEDO_BOUNDARIES and
                not math.isclose(self._albedo, 1.0)):
            string += '{0: <20}{1}{2}\n'.format('\tBoundary Albedo', '=\t', self._albedo)
        string += '{0: <20}'.format('\tCoefficients') + '\n'
        for coeff in self._coefficients:
            string += f'{coeff: <20}=\t{self._coefficients[coeff]}\n'
        return string

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            check_type('surface name', name, str)
            self._name = name
        else:
            self._name = ''

    @property
    def type(self):
        return self._type

    @property
    def boundary_type(self):
        return self._boundary_type

    @boundary_type.setter
    def boundary_type(self, boundary_type):
        check_type('boundary type', boundary_type, str)
        check_value('boundary type', boundary_type, _BOUNDARY_TYPES)
        self._boundary_type = boundary_type

    @property
    def albedo(self):
        return self._albedo

    @albedo.setter
    def albedo(self, albedo):
        check_type('albedo', albedo, Real)
        check_greater_than('albedo', albedo, 0.0)
        self._albedo = float(albedo)

    @property
    def coefficients(self):
        return self._coefficients

    def bounding_box(self, side):
        """Determine an axis-aligned bounding box.

        Parameters
        ----------
        side : {'+', '-'}
            Indicates the negative or positive half-space
        """
        return BoundingBox.infinite()

    def clone(self, memo=None):
        """Create a copy of this surface with a new unique ID."""
        if memo is None:
            memo = {}
        if self not in memo:
            clone = deepcopy(self)
            clone.id = None
            memo[self] = clone
        return memo[self]

    def normalize(self, coeffs=None):
        """Normalize coefficients by first nonzero value"""
        if coeffs is None:
            coeffs = self._get_base_coeffs()
        coeffs = np.asarray(coeffs)
        nonzeros = ~np.isclose(coeffs, 0., rtol=0., atol=self._atol)
        norm_factor = coeffs[nonzeros][0]
        return tuple([c/norm_factor for c in coeffs])

    def is_equal(self, other):
        """Determine if this Surface is equivalent to another"""
        coeffs1 = self.normalize(self._get_base_coeffs())
        coeffs2 = self.normalize(other._get_base_coeffs())
        return np.allclose(coeffs1, coeffs2, rtol=0., atol=self._atol)

    @abstractmethod
    def _get_base_coeffs(self):
        """Return polynomial coefficients representing the implicit surface
        equation."""

    @abstractmethod
    def evaluate(self, point):
        """Evaluate the surface equation at a given point."""

    @abstractmethod
    def translate(self, vector, inplace=False):
        """Translate surface in given direction"""

    @abstractmethod
    def rotate(self, rotation, pivot=(0., 0., 0.), order='xyz', inplace=False):
        """Rotate surface by angles provided or by applying matrix directly."""

    def to_xml_element(self):
        """Return XML representation of the surface"""
        element = ET.Element("surface")
        element.set("id", str(self._id))
        if len(self._name) > 0:
            element.set("name", str(self._name))
        element.set("type", self._type)
        if self.boundary_type != 'transmission':
            element.set("boundary", self.boundary_type)
            if (self.boundary_type in _ALBEDO_BOUNDARIES and
                    not math.isclose(self.albedo, 1.0)):
                element.set("albedo", str(self.albedo))
        element.set("coeffs", ' '.join([str(self._coefficients.setdefault(key, 0.0))
                                        for key in self._coeff_keys]))
        return element

    @staticmethod
    def from_xml_element(elem):
        """Generate surface from an XML element"""
        surf_type = get_text(elem, "type")
        cls = _SURFACE_CLASSES[surf_type]
        kwargs = {}
        kwargs['surface_id'] = int(get_text(elem, "id"))
        kwargs['boundary_type'] = get_text(elem, "boundary", "transmission")
        if kwargs['boundary_type'] in _ALBEDO_BOUNDARIES:
            kwargs['albedo'] = float(get_text(elem, "albedo", 1.0))
        kwargs['name'] = get_text(elem, "name")
        coeffs = get_elem_list(elem, "coeffs", float)
        kwargs.update(dict(zip(cls._coeff_keys, coeffs)))
        return cls(**kwargs)


class PlaneMixin:
    """A Plane mixin class for all operations on order 1 surfaces"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._periodic_surface = None

    @property
    def periodic_surface(self):
        return self._periodic_surface

    @periodic_surface.setter
    def periodic_surface(self, periodic_surface):
        check_type('periodic surface', periodic_surface, Plane)
        self._periodic_surface = periodic_surface
        periodic_surface._periodic_surface = self

    def _get_base_coeffs(self):
        return (self.a, self.b, self.c, self.d)

    def _get_normal(self):
        a, b, c = self._get_base_coeffs()[:3]
        return np.array((a, b, c)) / math.sqrt(a*a + b*b + c*c)

    @property
    def is_axial(self):
        """Whether the plane is perpendicular to the z-axis (a z-plane)."""
        a, b, _, _ = self._get_base_coeffs()
        return math.isclose(a, 0.0, abs_tol=self._atol) and \
            math.isclose(b, 0.0, abs_tol=self._atol)

    def bounding_box(self, side):
        """Determine an axis-aligned bounding box.

        An axis-aligned bounding box for Plane half-spaces is represented by
        its lower-left and upper-right coordinates. If the half-space is
        unbounded in a particular direction, numpy.inf is used to represent
        infinity.
        """
        nhat = self._get_normal()
        ll = np.array([-np.inf, -np.inf, -np.inf])
        ur = np.array([np.inf, np.inf, np.inf])
        if np.any(np.isclose(np.abs(nhat), 1., rtol=0., atol=self._atol)):
            sign = nhat.sum()
            a, b, c, d = self._get_base_coeffs()
            vals = [d/val if not np.isclose(val, 0., rtol=0., atol=self._atol)
                    else np.nan for val in (a, b, c)]
            if (side == '-') == (sign > 0):
                ur = np.array([v if not np.isnan(v) else np.inf for v in vals])
            else:
                ll = np.array([v if not np.isnan(v) else -np.inf for v in vals])
        return BoundingBox(ll, ur)

    def evaluate(self, point):
        """Evaluate the surface equation at a given point: Ax + By + Cz - D"""
        x, y, z = point
        a, b, c, d = self._get_base_coeffs()
        return a*x + b*y + c*z - d

    def translate(self, vector, inplace=False):
        """Translate surface in given direction"""
        if np.allclose(vector, 0., rtol=0., atol=self._atol):
            return self
        a, b, c, d = self._get_base_coeffs()
        d = d + np.dot([a, b, c], vector)
        surf = self if inplace else self.clone()
        setattr(surf, surf._coeff_keys[-1], float(d))
        return surf

    def rotate(self, rotation, pivot=(0., 0., 0.), order='xyz', inplace=False):
        """Rotate surface by angles provided or by applying matrix directly.
        Only rotations about the z-axis are supported."""
        pivot = np.asarray(pivot)
        Rmat = _rotation_matrix(rotation, order)
        _check_z_rotation(Rmat, f'{type(self).__name__}.rotate')
        surf = self.translate(-pivot, inplace=inplace)
        a, b, c, d = surf._get_base_coeffs()
        a, b, c = Rmat @ [a, b, c]
        kwargs = {'boundary_type': surf.boundary_type, 'albedo': surf.albedo,
                  'name': surf.name}
        if inplace:
            kwargs['surface_id'] = surf.id
        surf = Plane(a=float(a), b=float(b), c=float(c), d=float(d), **kwargs)
        return surf.translate(pivot, inplace=inplace)

    def to_xml_element(self):
        element = super().to_xml_element()
        if self.boundary_type == 'periodic' and self.periodic_surface is not None:
            element.set('periodic_surface_id', str(self.periodic_surface.id))
        return element


class Plane(PlaneMixin, Surface):
    """An arbitrary plane of the form :math:`Ax + By + Cz = D`.

    UM2 requires that the plane either contains the z-direction (c == 0) or is
    perpendicular to it (a == b == 0, only allowed as an axial bound).

    Parameters
    ----------
    a, b, c, d : float, optional
        The 'A', 'B', 'C' and 'D' parameters for the plane.
    boundary_type, albedo, name, surface_id
        See :class:`Surface`
    """

    _type = 'plane'
    _coeff_keys = ('a', 'b', 'c', 'd')

    def __init__(self, a=1., b=0., c=0., d=0., *args, **kwargs):
        kwargs = _future_kwargs_warning_helper(type(self), *args, **kwargs)
        capdict = {}
        for k in 'ABCD':
            val = kwargs.pop(k, None)
            if val is not None:
                warn(_WARNING_UPPER.format(type(self), k.lower(), k), FutureWarning)
                capdict[k.lower()] = val
        coeffs = dict(zip(self._coeff_keys, (a, b, c, d)))
        coeffs.update(capdict)
        a_, b_, c_ = coeffs['a'], coeffs['b'], coeffs['c']
        if (not math.isclose(c_, 0.0, abs_tol=self._atol)
                and not (math.isclose(a_, 0.0, abs_tol=self._atol)
                         and math.isclose(b_, 0.0, abs_tol=self._atol))):
            raise UM2NotSupportedError(
                'UM2 does not support planes that are oblique to the z-axis '
                f'(a={a_}, b={b_}, c={c_}), since the geometry must be Z-invariant.')
        super().__init__(**kwargs)
        for key, val in coeffs.items():
            setattr(self, key, val)

    @classmethod
    def __subclasshook__(cls, c):
        if cls is Plane and c in (XPlane, YPlane, ZPlane):
            return True
        return NotImplemented

    a = SurfaceCoefficient('a')
    b = SurfaceCoefficient('b')
    c = SurfaceCoefficient('c')
    d = SurfaceCoefficient('d')

    @classmethod
    def from_points(cls, p1, p2, p3, **kwargs):
        """Return a plane given three points that pass through it."""
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        p3 = np.asarray(p3, dtype=float)
        n = np.cross(p2 - p1, p3 - p1)
        if np.allclose(n, 0.):
            raise ValueError('All three points appear to lie along a line.')
        a, b, c = n
        d = np.dot(n, p1)
        return cls(a=float(a), b=float(b), c=float(c), d=float(d), **kwargs)

    def flip_normal(self):
        """Modify plane coefficients to reverse the normal vector."""
        self.a = -self.a
        self.b = -self.b
        self.c = -self.c
        self.d = -self.d


class XPlane(PlaneMixin, Surface):
    """A plane perpendicular to the x axis of the form :math:`x - x_0 = 0`

    Parameters
    ----------
    x0 : float, optional
        Location of the plane in [cm]. Defaults to 0.
    boundary_type, albedo, name, surface_id
        See :class:`Surface`
    """

    _type = 'x-plane'
    _coeff_keys = ('x0',)

    def __init__(self, x0=0., *args, **kwargs):
        kwargs = _future_kwargs_warning_helper(type(self), *args, **kwargs)
        super().__init__(**kwargs)
        self.x0 = x0

    x0 = SurfaceCoefficient('x0')
    a = SurfaceCoefficient(1.)
    b = SurfaceCoefficient(0.)
    c = SurfaceCoefficient(0.)
    d = x0

    def evaluate(self, point):
        return point[0] - self.x0


class YPlane(PlaneMixin, Surface):
    """A plane perpendicular to the y axis of the form :math:`y - y_0 = 0`

    Parameters
    ----------
    y0 : float, optional
        Location of the plane in [cm]. Defaults to 0.
    boundary_type, albedo, name, surface_id
        See :class:`Surface`
    """

    _type = 'y-plane'
    _coeff_keys = ('y0',)

    def __init__(self, y0=0., *args, **kwargs):
        kwargs = _future_kwargs_warning_helper(type(self), *args, **kwargs)
        super().__init__(**kwargs)
        self.y0 = y0

    y0 = SurfaceCoefficient('y0')
    a = SurfaceCoefficient(0.)
    b = SurfaceCoefficient(1.)
    c = SurfaceCoefficient(0.)
    d = y0

    def evaluate(self, point):
        return point[1] - self.y0


class ZPlane(PlaneMixin, Surface):
    """A plane perpendicular to the z axis of the form :math:`z - z_0 = 0`

    In UM2, z-planes may only be used to bound the entire model axially.

    Parameters
    ----------
    z0 : float, optional
        Location of the plane in [cm]. Defaults to 0.
    boundary_type, albedo, name, surface_id
        See :class:`Surface`
    """

    _type = 'z-plane'
    _coeff_keys = ('z0',)

    def __init__(self, z0=0., *args, **kwargs):
        kwargs = _future_kwargs_warning_helper(type(self), *args, **kwargs)
        super().__init__(**kwargs)
        self.z0 = z0

    z0 = SurfaceCoefficient('z0')
    a = SurfaceCoefficient(0.)
    b = SurfaceCoefficient(0.)
    c = SurfaceCoefficient(1.)
    d = z0

    def evaluate(self, point):
        return point[2] - self.z0


class QuadricMixin:
    """A Mixin class implementing common functionality for quadric surfaces"""

    def get_Abc(self, coeffs=None):
        """Compute matrix, vector, and scalar coefficients for this surface or
        for a specified set of coefficients."""
        if coeffs is None:
            a, b, c, d, e, f, g, h, j, k = self._get_base_coeffs()
        else:
            a, b, c, d, e, f, g, h, j, k = coeffs
        A = np.array([[a, d/2, f/2], [d/2, b, e/2], [f/2, e/2, c]])
        bvec = np.array([g, h, j])
        return A, bvec, k

    def eigh(self, coeffs=None):
        """Wrapper method for returning eigenvalues and eigenvectors of this
        quadric surface which is used for transformations."""
        return np.linalg.eigh(self.get_Abc(coeffs=coeffs)[0])

    def evaluate(self, point):
        x = np.asarray(point)
        A, b, c = self.get_Abc()
        return x.T @ A @ x + b.T @ x + c


class ZCylinder(QuadricMixin, Surface):
    """An infinite cylinder whose length is parallel to the z-axis of the form
    :math:`(x - x_0)^2 + (y - y_0)^2 - r^2 = 0`.

    Parameters
    ----------
    x0 : float, optional
        x-coordinate for the origin of the Cylinder in [cm]. Defaults to 0
    y0 : float, optional
        y-coordinate for the origin of the Cylinder in [cm]. Defaults to 0
    r : float, optional
        Radius of the cylinder in [cm]. Defaults to 1.
    boundary_type, albedo, name, surface_id
        See :class:`Surface`
    """

    _type = 'z-cylinder'
    _coeff_keys = ('x0', 'y0', 'r')

    def __init__(self, x0=0., y0=0., r=1., *args, **kwargs):
        R = kwargs.pop('R', None)
        if R is not None:
            warn(_WARNING_UPPER.format(type(self).__name__, 'r', 'R'), FutureWarning)
            r = R
        kwargs = _future_kwargs_warning_helper(type(self), *args, **kwargs)
        super().__init__(**kwargs)
        for key, val in zip(self._coeff_keys, (x0, y0, r)):
            setattr(self, key, val)

    x0 = SurfaceCoefficient('x0')
    y0 = SurfaceCoefficient('y0')
    z0 = SurfaceCoefficient(0.)
    r = SurfaceCoefficient('r', positive=True)
    dx = SurfaceCoefficient(0.)
    dy = SurfaceCoefficient(0.)
    dz = SurfaceCoefficient(1.)

    def _get_base_coeffs(self):
        x0, y0, r = self.x0, self.y0, self.r
        c = d = e = f = j = 0.
        a = b = 1.
        g, h, k = -2*x0, -2*y0, x0*x0 + y0*y0 - r*r
        return (a, b, c, d, e, f, g, h, j, k)

    def bounding_box(self, side):
        if side == '-':
            return BoundingBox(
                np.array([self.x0 - self.r, self.y0 - self.r, -np.inf]),
                np.array([self.x0 + self.r, self.y0 + self.r, np.inf]))
        return BoundingBox.infinite()

    def evaluate(self, point):
        x = point[0] - self.x0
        y = point[1] - self.y0
        return x*x + y*y - self.r**2

    def translate(self, vector, inplace=False):
        """Translate surface in given direction"""
        vector = np.asarray(vector, dtype=float)
        if np.allclose(vector, 0., rtol=0., atol=self._atol):
            return self
        surf = self if inplace else self.clone()
        surf.x0 = float(self.x0 + vector[0])
        surf.y0 = float(self.y0 + vector[1])
        return surf

    def rotate(self, rotation, pivot=(0., 0., 0.), order='xyz', inplace=False):
        """Rotate surface by angles provided or by applying matrix directly.
        Only rotations about the z-axis are supported."""
        pivot = np.asarray(pivot, dtype=float)
        Rmat = _rotation_matrix(rotation, order)
        _check_z_rotation(Rmat, 'ZCylinder.rotate')
        surf = self if inplace else self.clone()
        origin = Rmat @ (np.array([self.x0, self.y0, 0.]) - pivot) + pivot
        surf.x0 = float(origin[0])
        surf.y0 = float(origin[1])
        return surf


def _unsupported_surface(name, reason):
    """Create a class for an OpenMC surface that UM2 does not support."""

    def __init__(self, *args, **kwargs):
        raise UM2NotSupportedError(
            f'openmc.{name} is not supported by UM2: {reason}. UM2 generates 2D '
            'meshes of Z-invariant geometry; use ZCylinder, XPlane, YPlane or '
            'Plane (with c=0) instead.')

    doc = f"Not supported by UM2 ({reason}). Creating one raises UM2NotSupportedError."
    return type(name, (Surface,), {
        '__init__': __init__, '__doc__': doc, '_um2_unsupported': True,
        '_get_base_coeffs': lambda self: None,
        'evaluate': lambda self, point: None,
        'translate': lambda self, vector, inplace=False: None,
        'rotate': lambda self, rotation, pivot=(0., 0., 0.), order='xyz',
        inplace=False: None,
    })


Cylinder = _unsupported_surface('Cylinder', 'arbitrary cylinders are not Z-invariant')
XCylinder = _unsupported_surface('XCylinder', 'x-cylinders are not Z-invariant')
YCylinder = _unsupported_surface('YCylinder', 'y-cylinders are not Z-invariant')
Sphere = _unsupported_surface('Sphere', 'spheres are not Z-invariant')
Cone = _unsupported_surface('Cone', 'cones are not Z-invariant')
XCone = _unsupported_surface('XCone', 'cones are not Z-invariant')
YCone = _unsupported_surface('YCone', 'cones are not Z-invariant')
ZCone = _unsupported_surface('ZCone', 'cones are not Z-invariant')
Quadric = _unsupported_surface('Quadric', 'general quadrics are not supported')
XTorus = _unsupported_surface('XTorus', 'tori are not Z-invariant')
YTorus = _unsupported_surface('YTorus', 'tori are not Z-invariant')
ZTorus = _unsupported_surface('ZTorus', 'tori are not Z-invariant')


class Halfspace(Region):
    """A positive or negative half-space region.

    A half-space is either of the two parts into which a two-dimension surface
    divides the three-dimensional Euclidean space. Instances of Halfspace are
    generally not instantiated directly but rather are created with the unary
    operators - and + applied to a :class:`Surface`.

    Parameters
    ----------
    surface : openmc.Surface
        Surface which divides Euclidean space.
    side : {'+', '-'}
        Indicates whether the positive or negative half-space is used.
    """

    def __init__(self, surface, side):
        self.surface = surface
        self.side = side

    def __and__(self, other):
        if isinstance(other, Intersection):
            return Intersection([self] + other[:])
        return Intersection((self, other))

    def __or__(self, other):
        if isinstance(other, Union):
            return Union([self] + other[:])
        return Union((self, other))

    def __invert__(self) -> Halfspace:
        return -self.surface if self.side == '+' else +self.surface

    def __contains__(self, point):
        val = self.surface.evaluate(point)
        return val >= 0. if self.side == '+' else val < 0.

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, surface):
        check_type('surface', surface, Surface)
        self._surface = surface

    @property
    def side(self):
        return self._side

    @side.setter
    def side(self, side):
        check_value('side', side, ('+', '-'))
        self._side = side

    @property
    def bounding_box(self):
        return self.surface.bounding_box(self.side)

    def __str__(self):
        return '-' + str(self.surface.id) if self.side == '-' else str(self.surface.id)

    def get_surfaces(self, surfaces=None):
        """Returns the surface that this is a halfspace of."""
        if surfaces is None:
            surfaces = {}
        surfaces[self.surface.id] = self.surface
        return surfaces

    def remove_redundant_surfaces(self, redundant_surfaces):
        """Recursively remove all redundant surfaces referenced by this region"""
        surf = redundant_surfaces.get(self.surface.id)
        if surf is not None:
            self.surface = surf

    def clone(self, memo=None):
        """Create a copy of this halfspace, with a cloned surface with a
        unique ID."""
        if memo is None:
            memo = {}
        clone = deepcopy(self)
        clone.surface = self.surface.clone(memo)
        return clone

    def translate(self, vector, inplace=False, memo=None):
        """Translate half-space in given direction"""
        if memo is None:
            memo = {}
        key = (self.surface, tuple(vector))
        if key not in memo:
            memo[key] = self.surface.translate(vector, inplace)
        return type(self)(memo[key], self.side)

    def rotate(self, rotation, pivot=(0., 0., 0.), order='xyz', inplace=False,
               memo=None):
        r"""Rotate surface by angles provided or by applying matrix directly."""
        if memo is None:
            memo = {}
        key = (self.surface, tuple(np.ravel(rotation)), tuple(pivot), order, inplace)
        if key not in memo:
            memo[key] = self.surface.rotate(rotation, pivot=pivot, order=order,
                                            inplace=inplace)
        return type(self)(memo[key], self.side)


_SURFACE_CLASSES = {cls._type: cls for cls in (Plane, XPlane, YPlane, ZPlane, ZCylinder)}
