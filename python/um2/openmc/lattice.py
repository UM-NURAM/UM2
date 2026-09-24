"""Lattices (compatible with :class:`openmc.RectLattice` and
:class:`openmc.HexLattice`).

UM2 only supports 2D lattices: a RectLattice with a 2D pitch and a HexLattice
without axial levels.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from copy import deepcopy
from math import floor, sqrt
from numbers import Real
import xml.etree.ElementTree as ET

import numpy as np

from ..errors import UM2NotSupportedError
from . import checkvalue as cv
from ._xml import get_elem_list, get_text
from .mixin import IDManagerMixin
from .universe import UniverseBase


class Lattice(IDManagerMixin, ABC):
    """A repeating structure wherein each element is a universe.

    Parameters
    ----------
    lattice_id : int, optional
        Unique identifier for the lattice. If not specified, an identifier will
        automatically be assigned.
    name : str, optional
        Name of the lattice. If not specified, the name is the empty string.
    """

    next_id = 1
    used_ids = UniverseBase.used_ids

    def __init__(self, lattice_id=None, name=''):
        self.id = lattice_id
        self.name = name
        self._pitch = None
        self._outer = None
        self._universes = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            cv.check_type('lattice name', name, str)
            self._name = name
        else:
            self._name = ''

    @property
    def pitch(self):
        return self._pitch

    @property
    def outer(self):
        return self._outer

    @outer.setter
    def outer(self, outer):
        cv.check_type('outer universe', outer, UniverseBase)
        self._outer = outer

    @property
    def universes(self):
        return self._universes

    def get_unique_universes(self):
        """Determine all unique universes in the lattice

        Returns
        -------
        universes : dict
            Dictionary whose keys are universe IDs and values are
            :class:`openmc.UniverseBase` instances
        """
        univs = {}
        for k in range(len(self._universes)):
            for j in range(len(self._universes[k])):
                u = self._universes[k][j]
                univs[u._id] = u
        if self.outer is not None:
            univs[self.outer._id] = self.outer
        return univs

    def get_nuclides(self):
        """Returns all nuclides in the lattice"""
        nuclides = []
        for universe in self.get_unique_universes().values():
            for nuclide in universe.get_nuclides():
                if nuclide not in nuclides:
                    nuclides.append(nuclide)
        return nuclides

    def get_all_cells(self, memo=None):
        """Return all cells that are contained within the lattice"""
        cells = {}
        if memo is None:
            memo = set()
        elif self in memo:
            return cells
        memo.add(self)
        for universe in self.get_unique_universes().values():
            cells.update(universe.get_all_cells(memo))
        return cells

    def get_all_materials(self, memo=None):
        """Return all materials that are contained within the lattice"""
        if memo is None:
            memo = set()
        materials = {}
        for cell in self.get_all_cells(memo).values():
            materials.update(cell.get_all_materials(memo))
        return materials

    def get_all_universes(self, memo=None):
        """Return all universes that are contained within the lattice"""
        all_universes = {}
        if memo is None:
            memo = set()
        elif self in memo:
            return all_universes
        memo.add(self)
        unique_universes = self.get_unique_universes()
        all_universes.update(unique_universes)
        for universe in unique_universes.values():
            all_universes.update(universe.get_all_universes(memo))
        return all_universes

    def get_universe(self, idx):
        """Return universe corresponding to a lattice element index

        Parameters
        ----------
        idx : Iterable of int
            Lattice element indices in the :math:`(x,y)` coordinate system for
            rectangular lattices and :math:`(x,\\alpha)` (or
            :math:`(\\alpha,y)`) for hexagonal lattices.
        """
        idx_u = self.get_universe_index(idx)
        return self.universes[idx_u[0]][idx_u[1]]

    def find(self, point):
        """Find cells/universes/lattices which contain a given point"""
        idx, p = self.find_element(point)
        if self.is_valid_index(idx):
            u = self.get_universe(idx)
        elif self.outer is not None:
            u = self.outer
        else:
            return []
        return [(self, idx)] + u.find(p)

    def clone(self, clone_materials=True, clone_regions=True, memo=None):
        """Create a copy of this lattice with a new unique ID, and clones all
        universes within this lattice."""
        if memo is None:
            memo = {}
        if self not in memo:
            clone = deepcopy(self)
            clone.id = None
            if self.outer is not None:
                clone.outer = self.outer.clone(clone_materials, clone_regions, memo)
            for i in self.indices:
                u = self.universes[i[0]][i[1]]
                clone.universes[i[0]][i[1]] = u.clone(clone_materials, clone_regions, memo)
            memo[self] = clone
        return memo[self]


class RectLattice(Lattice):
    """A lattice consisting of rectangular prisms.

    To completely define a rectangular lattice, the :attr:`RectLattice.lower_left`
    :attr:`RectLattice.pitch`, :attr:`RectLattice.outer`, and
    :attr:`RectLattice.universes` properties need to be set.

    The universes array is indexed ``[row][column]`` where row 0 is the top of
    the lattice (largest y), as in OpenMC.

    Parameters
    ----------
    lattice_id : int, optional
        Unique identifier for the lattice. If not specified, an identifier will
        automatically be assigned.
    name : str, optional
        Name of the lattice. If not specified, the name is the empty string.
    """

    def __init__(self, lattice_id=None, name=''):
        super().__init__(lattice_id, name)
        self._lower_left = None

    def __repr__(self):
        string = 'RectLattice\n'
        string += '{: <16}=\t{}\n'.format('\tID', self._id)
        string += '{: <16}=\t{}\n'.format('\tName', self._name)
        string += '{: <16}=\t{}\n'.format('\tShape', self.shape)
        string += '{: <16}=\t{}\n'.format('\tLower Left', self._lower_left)
        string += '{: <16}=\t{}\n'.format('\tPitch', self._pitch)
        string += '{: <16}=\t{}\n'.format(
            '\tOuter', self._outer._id if self._outer is not None else None)
        string += '{: <16}\n'.format('\tUniverses')
        for i, universe in enumerate(np.ravel(self._universes)):
            string += f'{universe._id} '
            if (i + 1) % self.shape[0] == 0:
                string += '\n'
        return string.rstrip('\n')

    @property
    def indices(self):
        return list(np.broadcast(*np.ogrid[:self.shape[1], :self.shape[0]]))

    @property
    def _natural_indices(self):
        nx, ny = self.shape
        for iy in range(ny):
            for ix in range(nx):
                yield (ix, iy)

    @property
    def lower_left(self):
        return self._lower_left

    @lower_left.setter
    def lower_left(self, lower_left):
        cv.check_type('lattice lower left corner', lower_left, Iterable, Real)
        cv.check_length('lattice lower left corner', lower_left, 2, 3)
        self._lower_left = lower_left

    @property
    def ndim(self):
        if self.pitch is not None:
            return len(self.pitch)
        raise ValueError('Number of dimensions cannot be determined until '
                         'the lattice pitch has been set.')

    @property
    def shape(self):
        return self._universes.shape[::-1]

    @Lattice.pitch.setter
    def pitch(self, pitch):
        cv.check_type('lattice pitch', pitch, Iterable, Real)
        cv.check_length('lattice pitch', pitch, 2, 3)
        for dim in pitch:
            cv.check_greater_than('lattice pitch', dim, 0.0)
        if len(pitch) == 3:
            raise UM2NotSupportedError(
                f'RectLattice ID="{self._id}": 3D lattices (with an axial pitch) are '
                'not supported by UM2.')
        self._pitch = pitch

    @Lattice.universes.setter
    def universes(self, universes):
        cv.check_iterable_type('lattice universes', universes, UniverseBase,
                               min_depth=2, max_depth=3)
        array = np.empty((len(universes), len(universes[0])), dtype=object)
        for j, row in enumerate(universes):
            if isinstance(row, UniverseBase) or len(row) != array.shape[1]:
                raise ValueError('lattice universes must be a rectangular 2D array')
            for i, u in enumerate(row):
                if not isinstance(u, UniverseBase):
                    raise UM2NotSupportedError(
                        f'RectLattice ID="{self._id}": 3D lattices are not '
                        'supported by UM2.')
                array[j, i] = u
        self._universes = array

    def find_element(self, point):
        """Determine index of lattice element and local coordinates for a point"""
        ix = floor((point[0] - self.lower_left[0])/self.pitch[0])
        iy = floor((point[1] - self.lower_left[1])/self.pitch[1])
        idx = (ix, iy)
        return idx, self.get_local_coordinates(point, idx)

    def get_local_coordinates(self, point, idx):
        """Determine local coordinates of a point within a lattice element"""
        x = point[0] - (self.lower_left[0] + (idx[0] + 0.5)*self.pitch[0])
        y = point[1] - (self.lower_left[1] + (idx[1] + 0.5)*self.pitch[1])
        return (x, y, point[2])

    def get_universe_index(self, idx):
        """Return index in the universes array corresponding to a lattice
        element index (ix, iy)"""
        max_y = self.shape[1] - 1
        x, y = idx[0], idx[1]
        return (max_y - y, x)

    def is_valid_index(self, idx):
        """Determine whether lattice element index is within defined range"""
        return 0 <= idx[0] < self.shape[0] and 0 <= idx[1] < self.shape[1]

    def element_center(self, idx):
        """Return the (x, y) center of lattice element (ix, iy)."""
        return (self.lower_left[0] + (idx[0] + 0.5)*self.pitch[0],
                self.lower_left[1] + (idx[1] + 0.5)*self.pitch[1])

    def create_xml_subelement(self, xml_element, memo=None):
        if memo is None:
            memo = set()
        elif self in memo:
            return
        memo.add(self)
        if self.universes is None:
            raise ValueError(f'Lattice {self.id} does not have universes assigned.')

        lattice_subelement = ET.Element("lattice")
        lattice_subelement.set("id", str(self._id))
        if len(self._name) > 0:
            lattice_subelement.set("name", str(self._name))
        pitch = ET.SubElement(lattice_subelement, "pitch")
        pitch.text = ' '.join(map(str, self._pitch))
        if self._outer is not None:
            outer = ET.SubElement(lattice_subelement, "outer")
            outer.text = str(self._outer._id)
            self._outer.create_xml_subelement(xml_element, memo)
        dimension = ET.SubElement(lattice_subelement, "dimension")
        dimension.text = ' '.join(map(str, self.shape))
        if self.lower_left is None:
            raise ValueError(f'Lattice {self.id} does not have lower_left specified.')
        lower_left = ET.SubElement(lattice_subelement, "lower_left")
        lower_left.text = ' '.join(map(str, self._lower_left))

        universe_ids = '\n'
        for y in range(self.shape[1]):
            for x in range(self.shape[0]):
                universe = self._universes[y][x]
                universe_ids += f'{universe._id} '
                universe.create_xml_subelement(xml_element, memo)
            universe_ids += '\n'
        universes = ET.SubElement(lattice_subelement, "universes")
        universes.text = universe_ids.rstrip('\n')
        xml_element.append(lattice_subelement)

    @classmethod
    def from_xml_element(cls, elem, get_universe):
        """Generate rectangular lattice from XML element"""
        lat = cls(int(get_text(elem, 'id')), get_text(elem, 'name'))
        lat.lower_left = get_elem_list(elem, "lower_left", float)
        lat.pitch = get_elem_list(elem, "pitch", float)
        outer = get_text(elem, 'outer')
        if outer is not None:
            lat.outer = get_universe(int(outer))
        nx, ny = get_elem_list(elem, "dimension", int)
        ids = get_elem_list(elem, "universes", int)
        lat.universes = [[get_universe(ids[j*nx + i]) for i in range(nx)]
                         for j in range(ny)]
        return lat


class HexLattice(Lattice):
    r"""A lattice consisting of hexagonal prisms.

    To completely define a hexagonal lattice, the :attr:`HexLattice.center`,
    :attr:`HexLattice.pitch`, :attr:`HexLattice.universes`, and
    :attr:`HexLattice.outer` properties need to be set.

    Universes are given as a list of rings, from the outermost ring to the
    innermost element. Each ring starts at the "top" and proceeds clockwise.

    Parameters
    ----------
    lattice_id : int, optional
        Unique identifier for the lattice. If not specified, an identifier will
        automatically be assigned.
    name : str, optional
        Name of the lattice. If not specified, the name is the empty string.
    """

    def __init__(self, lattice_id=None, name=''):
        super().__init__(lattice_id, name)
        self._num_rings = None
        self._num_axial = None
        self._center = None
        self._orientation = 'y'

    def __repr__(self):
        string = 'HexLattice\n'
        string += '{0: <16}{1}{2}\n'.format('\tID', '=\t', self._id)
        string += '{0: <16}{1}{2}\n'.format('\tName', '=\t', self._name)
        string += '{0: <16}{1}{2}\n'.format('\tOrientation', '=\t', self._orientation)
        string += '{0: <16}{1}{2}\n'.format('\t# Rings', '=\t', self._num_rings)
        string += '{0: <16}{1}{2}\n'.format('\t# Axial', '=\t', self._num_axial)
        string += '{0: <16}{1}{2}\n'.format('\tCenter', '=\t', self._center)
        string += '{0: <16}{1}{2}\n'.format('\tPitch', '=\t', self._pitch)
        outer = self._outer._id if self._outer is not None else None
        string += '{0: <16}{1}{2}\n'.format('\tOuter', '=\t', outer)
        string += '{0: <16}\n'.format('\tUniverses')
        string += self._repr_axial_slice(self._universes)
        return string

    @property
    def num_rings(self):
        return self._num_rings

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        cv.check_value('orientation', orientation.lower(), ('x', 'y'))
        self._orientation = orientation.lower()

    @property
    def num_axial(self):
        return self._num_axial

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        cv.check_type('lattice center', center, Iterable, Real)
        cv.check_length('lattice center', center, 2, 3)
        self._center = center

    @property
    def indices(self):
        return [(r, i) for r in range(self.num_rings)
                for i in range(max(6*(self.num_rings - 1 - r), 1))]

    @property
    def _natural_indices(self):
        r = self.num_rings
        for a in range(-r + 1, r):
            for x in range(-r + 1, r):
                idx = (x, a)
                if self.is_valid_index(idx):
                    yield idx

    @property
    def ndim(self):
        return 2

    @Lattice.pitch.setter
    def pitch(self, pitch):
        cv.check_type('lattice pitch', pitch, Iterable, Real)
        cv.check_length('lattice pitch', pitch, 1, 2)
        for dim in pitch:
            cv.check_greater_than('lattice pitch', dim, 0)
        if len(pitch) == 2:
            raise UM2NotSupportedError(
                f'HexLattice ID="{self._id}": 3D lattices (with an axial pitch) are '
                'not supported by UM2.')
        self._pitch = pitch

    @Lattice.universes.setter
    def universes(self, universes):
        cv.check_iterable_type('lattice universes', universes, UniverseBase,
                               min_depth=2, max_depth=3)
        if not isinstance(universes[0][0], UniverseBase):
            raise UM2NotSupportedError(
                f'HexLattice ID="{self._id}": 3D lattices (with axial levels) are '
                'not supported by UM2.')
        num_rings = len(universes)
        if len(universes[-1]) != 1:
            raise ValueError(f'HexLattice ID={self._id:d} has the wrong number of '
                             'elements in the innermost ring.  Only 1 element is '
                             'allowed in the innermost ring.')
        for r in range(num_rings - 1):
            if len(universes[r]) != 6*(num_rings - 1 - r):
                raise ValueError(
                    f'HexLattice ID={self._id:d} has the wrong number of elements in '
                    f'ring number {r:d} (counting from the outermost ring).  This '
                    f'ring should have {6*(num_rings - 1 - r):d} elements.')
        self._universes = universes
        self._num_axial = None
        self._num_rings = num_rings

    def find_element(self, point):
        r"""Determine index of lattice element and local coordinates for a point"""
        x = point[0] - self.center[0]
        y = point[1] - self.center[1]
        if self._orientation == 'x':
            alpha = y - x*sqrt(3.)
            i1 = floor(-alpha/(sqrt(3.0) * self.pitch[0]))
            i2 = floor(y/(sqrt(0.75) * self.pitch[0]))
        else:
            alpha = y - x/sqrt(3.)
            i1 = floor(x/(sqrt(0.75) * self.pitch[0]))
            i2 = floor(alpha/self.pitch[0])
        indices = [(i1, i2), (i1 + 1, i2), (i1, i2 + 1), (i1 + 1, i2 + 1)]
        d_min = np.inf
        for idx in indices:
            p = self.get_local_coordinates(point, idx)
            d = p[0]**2 + p[1]**2
            if d < d_min:
                d_min = d
                idx_min = idx
                p_min = p
        return idx_min, p_min

    def element_center(self, idx):
        """Return the (x, y) center of lattice element with natural index idx."""
        p = self.pitch[0]
        if self._orientation == 'x':
            return (self.center[0] + (idx[0] + 0.5*idx[1])*p,
                    self.center[1] + sqrt(0.75)*p*idx[1])
        return (self.center[0] + sqrt(0.75)*p*idx[0],
                self.center[1] + (0.5*idx[0] + idx[1])*p)

    def get_local_coordinates(self, point, idx):
        r"""Determine local coordinates of a point within a lattice element"""
        cx, cy = self.element_center(idx)
        return (point[0] - cx, point[1] - cy, point[2])

    def get_universe_index(self, idx):
        r"""Return index in the universes array corresponding to a lattice
        element index (natural indices)"""
        x = idx[0]
        a = idx[1]
        z = -a - x
        g = max(abs(x), abs(a), abs(z))
        i_ring = self._num_rings - 1 - g
        if x >= 0:
            if a >= 0:
                i_within = x
            else:
                i_within = 2*g + z
        else:
            if a <= 0:
                i_within = 3*g - x
            else:
                i_within = 5*g - z
        if self._orientation == 'x' and g > 0:
            i_within = (i_within + 5*g) % (6*g)
        return (i_ring, i_within)

    def is_valid_index(self, idx):
        r"""Determine whether lattice element index is within defined range"""
        x = idx[0]
        y = idx[1]
        z = 0 - y - x
        g = max(abs(x), abs(y), abs(z))
        return g < self.num_rings

    def create_xml_subelement(self, xml_element, memo=None):
        if memo is None:
            memo = set()
        elif self in memo:
            return
        memo.add(self)

        lattice_subelement = ET.Element("hex_lattice")
        lattice_subelement.set("id", str(self._id))
        if len(self._name) > 0:
            lattice_subelement.set("name", str(self._name))
        pitch = ET.SubElement(lattice_subelement, "pitch")
        pitch.text = ' '.join(map(str, self._pitch))
        if self._outer is not None:
            outer = ET.SubElement(lattice_subelement, "outer")
            outer.text = str(self._outer._id)
            self._outer.create_xml_subelement(xml_element, memo)
        lattice_subelement.set("n_rings", str(self._num_rings))
        if self._orientation == 'x':
            lattice_subelement.set("orientation", "x")
        center = ET.SubElement(lattice_subelement, "center")
        center.text = ' '.join(map(str, self._center))
        if self.universes is None:
            raise ValueError(f'Lattice {self.id} does not have universes assigned.')

        self._universes[-1][0].create_xml_subelement(xml_element, memo)
        for r in range(self._num_rings - 1):
            for theta in range(6*(self._num_rings - 1 - r)):
                self._universes[r][theta].create_xml_subelement(xml_element, memo)
        universes = ET.SubElement(lattice_subelement, "universes")
        universes.text = '\n' + self._repr_axial_slice(self._universes)
        xml_element.append(lattice_subelement)

    def _repr_axial_slice(self, universes):
        if self._orientation == 'x':
            return self._repr_axial_slice_x(universes)
        return self._repr_axial_slice_y(universes)

    def _repr_axial_slice_x(self, universes):
        largest_id = max(max(univ._id for univ in ring) for ring in universes)
        n_digits = len(str(largest_id))
        pad = ' '*n_digits
        id_form = '{: ^' + str(n_digits) + 'd}'
        rows = [[] for i in range(2*self._num_rings - 1)]
        middle = self._num_rings - 1
        rows[middle] = [id_form.format(universes[-1][0]._id)]
        for r in range(1, self._num_rings):
            r_prime = self._num_rings - 1 - r
            theta = 0
            y = middle
            for _ in range(r):
                rows[y].append(id_form.format(universes[r_prime][theta]._id))
                y += 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, id_form.format(universes[r_prime][theta]._id))
                theta += 1
            for _ in range(r):
                rows[y].insert(0, id_form.format(universes[r_prime][theta]._id))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, id_form.format(universes[r_prime][theta]._id))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].append(id_form.format(universes[r_prime][theta]._id))
                theta += 1
            for _ in range(r):
                rows[y].append(id_form.format(universes[r_prime][theta]._id))
                y += 1
                theta += 1
        rows = [pad.join(x) for x in rows]
        for y in range(self._num_rings - 1):
            rows[y] = (self._num_rings - 1 - y)*pad + rows[y]
            rows[-1 - y] = (self._num_rings - 1 - y)*pad + rows[-1 - y]
        return '\n'.join(rows)

    def _repr_axial_slice_y(self, universes):
        largest_id = max(max(univ._id for univ in ring) for ring in universes)
        n_digits = len(str(largest_id))
        pad = ' '*n_digits
        id_form = '{: ^' + str(n_digits) + 'd}'
        rows = [[] for i in range(1 + 4 * (self._num_rings-1))]
        middle = 2 * (self._num_rings - 1)
        rows[middle] = [id_form.format(universes[-1][0]._id)]
        for r in range(1, self._num_rings):
            r_prime = self._num_rings - 1 - r
            theta = 0
            y = middle + 2*r
            for _ in range(r):
                rows[y].append(id_form.format(universes[r_prime][theta]._id))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].append(id_form.format(universes[r_prime][theta]._id))
                y -= 2
                theta += 1
            for _ in range(r):
                rows[y].append(id_form.format(universes[r_prime][theta]._id))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, id_form.format(universes[r_prime][theta]._id))
                y += 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, id_form.format(universes[r_prime][theta]._id))
                y += 2
                theta += 1
            for _ in range(r):
                rows[y].insert(0, id_form.format(universes[r_prime][theta]._id))
                y += 1
                theta += 1
        rows = [pad.join(x) for x in rows[::-1]]
        for y in range(self._num_rings - 1):
            rows[y] = (self._num_rings - 1 - y)*pad + rows[y]
            rows[-1 - y] = (self._num_rings - 1 - y)*pad + rows[-1 - y]
        for y in range(self._num_rings % 2, self._num_rings, 2):
            rows[middle + y] = pad + rows[middle + y]
            if y != 0:
                rows[middle - y] = pad + rows[middle - y]
        return '\n'.join(rows)

    @staticmethod
    def _show_indices_y(num_rings):
        largest_index = 6*(num_rings - 1)
        n_digits_index = len(str(largest_index))
        n_digits_ring = len(str(num_rings - 1))
        str_form = f'({{:{n_digits_ring}}},{{:{n_digits_index}}})'
        pad = ' '*(n_digits_index + n_digits_ring + 3)
        rows = [[] for i in range(1 + 4 * (num_rings-1))]
        middle = 2 * (num_rings - 1)
        rows[middle] = [str_form.format(num_rings - 1, 0)]
        for r in range(1, num_rings):
            r_prime = num_rings - 1 - r
            theta = 0
            y = middle + 2*r
            for _ in range(r):
                rows[y].append(str_form.format(r_prime, theta))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].append(str_form.format(r_prime, theta))
                y -= 2
                theta += 1
            for _ in range(r):
                rows[y].append(str_form.format(r_prime, theta))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, str_form.format(r_prime, theta))
                y += 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, str_form.format(r_prime, theta))
                y += 2
                theta += 1
            for _ in range(r):
                rows[y].insert(0, str_form.format(r_prime, theta))
                y += 1
                theta += 1
        rows = [pad.join(x) for x in rows[::-1]]
        for y in range(num_rings - 1):
            rows[y] = (num_rings - 1 - y)*pad + rows[y]
            rows[-1 - y] = (num_rings - 1 - y)*pad + rows[-1 - y]
        for y in range(num_rings % 2, num_rings, 2):
            rows[middle + y] = pad + rows[middle + y]
            if y != 0:
                rows[middle - y] = pad + rows[middle - y]
        return '\n'.join(rows)

    @staticmethod
    def _show_indices_x(num_rings):
        largest_index = 6*(num_rings - 1)
        n_digits_index = len(str(largest_index))
        n_digits_ring = len(str(num_rings - 1))
        str_form = f'({{:{n_digits_ring}}},{{:{n_digits_index}}})'
        pad = ' '*(n_digits_index + n_digits_ring + 3)
        rows = [[] for i in range(2*num_rings - 1)]
        middle = num_rings - 1
        rows[middle] = [str_form.format(num_rings - 1, 0)]
        for r in range(1, num_rings):
            r_prime = num_rings - 1 - r
            theta = 0
            y = middle
            for _ in range(r):
                rows[y].append(str_form.format(r_prime, theta))
                y += 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, str_form.format(r_prime, theta))
                theta += 1
            for _ in range(r):
                rows[y].insert(0, str_form.format(r_prime, theta))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].insert(0, str_form.format(r_prime, theta))
                y -= 1
                theta += 1
            for _ in range(r):
                rows[y].append(str_form.format(r_prime, theta))
                theta += 1
            for _ in range(r):
                rows[y].append(str_form.format(r_prime, theta))
                y += 1
                theta += 1
        rows = [pad.join(x) for x in rows]
        for y in range(num_rings - 1):
            rows[y] = (num_rings - 1 - y)*pad + rows[y]
            rows[-1 - y] = (num_rings - 1 - y)*pad + rows[-1 - y]
        return '\n\n'.join(rows)

    @staticmethod
    def show_indices(num_rings, orientation='y'):
        """Return a diagram of the hexagonal lattice layout with indices.

        Parameters
        ----------
        num_rings : int
            Number of rings in the hexagonal lattice
        orientation : {"x", "y"}
            Orientation of the hexagonal lattice

        Returns
        -------
        str
            Diagram of the hexagonal lattice showing indices
        """
        if orientation == 'x':
            return HexLattice._show_indices_x(num_rings)
        return HexLattice._show_indices_y(num_rings)
