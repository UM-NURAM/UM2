"""Cells (compatible with :class:`openmc.Cell`)."""

from __future__ import annotations

from collections.abc import Iterable
from math import cos, pi, sin
from numbers import Real
import xml.etree.ElementTree as ET

import numpy as np

from ..errors import UM2NotSupportedError
from . import checkvalue as cv
from .bounding_box import BoundingBox
from .material import Material
from .mixin import IDManagerMixin
from .region import Complement, Region
from .surface import Halfspace


class Cell(IDManagerMixin):
    r"""A region of space defined as the intersection of half-space created by
    quadric surfaces.

    Parameters
    ----------
    cell_id : int, optional
        Unique identifier for the cell. If not specified, an identifier will
        automatically be assigned.
    name : str, optional
        Name of the cell. If not specified, the name is the empty string.
    fill : openmc.Material or openmc.UniverseBase or openmc.Lattice or None
        Indicates what the region of space is filled with
    region : openmc.Region, optional
        Region of space that is assigned to the cell.

    Notes
    -----
    UM2 does not support distributed material fills (a list of materials), and
    only supports rotations about the z-axis.
    """

    next_id = 1
    used_ids = set()

    def __init__(self, cell_id=None, name='', fill=None, region=None):
        self.id = cell_id
        self.name = name
        self.fill = fill
        self.region = region
        self._rotation = None
        self._rotation_matrix = None
        self._temperature = None
        self._density = None
        self._translation = None
        self._volume = None

    def __contains__(self, point):
        if self.region is None:
            return True
        return point in self.region

    def __repr__(self):
        string = 'Cell\n'
        string += '{: <16}=\t{}\n'.format('\tID', self.id)
        string += '{: <16}=\t{}\n'.format('\tName', self.name)
        if self.fill_type == 'material':
            string += '{: <16}=\tMaterial {}\n'.format('\tFill', self.fill.id)
        elif self.fill_type == 'void':
            string += '{: <16}=\tNone\n'.format('\tFill')
        else:
            string += '{: <16}=\t{}\n'.format('\tFill', self.fill.id)
        string += '{: <16}=\t{}\n'.format('\tRegion', self.region)
        string += '{: <16}=\t{}\n'.format('\tRotation', self.rotation)
        if self.fill_type == 'material':
            string += '\t{0: <15}=\t{1}\n'.format('Temperature', self.temperature)
            string += '\t{0: <15}=\t{1}\n'.format('Density', self.density)
        string += '{: <16}=\t{}\n'.format('\tTranslation', self.translation)
        string += '{: <16}=\t{}\n'.format('\tVolume', self.volume)
        return string

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            cv.check_type('cell name', name, str)
            self._name = name
        else:
            self._name = ''

    @property
    def fill(self):
        return self._fill

    @fill.setter
    def fill(self, fill):
        from .lattice import Lattice
        from .universe import UniverseBase
        if fill is not None:
            if isinstance(fill, Iterable) and not isinstance(fill, str):
                raise UM2NotSupportedError(
                    f'Cell ID="{self._id}": distributed material fills (a list '
                    'of materials) are not supported by UM2.')
            if not isinstance(fill, (Material, Lattice, UniverseBase)):
                raise ValueError(f'Unable to set Cell ID="{self._id}" to use a '
                                 f'non-Material or Universe fill "{fill}"')
        self._fill = fill

    @property
    def fill_type(self):
        from .lattice import Lattice
        from .universe import UniverseBase
        if isinstance(self.fill, Material):
            return 'material'
        if isinstance(self.fill, UniverseBase):
            return 'universe'
        if isinstance(self.fill, Lattice):
            return 'lattice'
        return 'void'

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region):
        if region is not None:
            cv.check_type('cell region', region, Region)
        self._region = region

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        cv.check_length('cell rotation', rotation, 3)
        rotation = np.asarray(rotation, dtype=float)
        if rotation.ndim == 2:
            matrix = rotation
        else:
            phi, theta, psi = rotation*(-pi/180.)
            c3, s3 = cos(phi), sin(phi)
            c2, s2 = cos(theta), sin(theta)
            c1, s1 = cos(psi), sin(psi)
            matrix = np.array([
                [c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
                [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3],
                [-s2, c2*s3, c2*c3]])
        if not (np.allclose(matrix[2], (0., 0., 1.), atol=1e-12)
                and np.allclose(matrix[:, 2], (0., 0., 1.), atol=1e-12)):
            raise UM2NotSupportedError(
                f'Cell ID="{self._id}": UM2 only supports rotations about the '
                'z-axis, since the geometry must be Z-invariant.')
        self._rotation = rotation
        self._rotation_matrix = matrix

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        cv.check_type('cell temperature', temperature, (Iterable, Real), none_ok=True)
        if isinstance(temperature, Iterable):
            cv.check_type('cell temperature', temperature, Iterable, Real)
            for T in temperature:
                cv.check_greater_than('cell temperature', T, 0.0, True)
        elif isinstance(temperature, Real):
            cv.check_greater_than('cell temperature', temperature, 0.0, True)
        if self.fill_type in ('universe', 'lattice'):
            for c in self.get_all_cells().values():
                if c.fill_type == 'material':
                    c._temperature = temperature
        else:
            self._temperature = temperature

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        cv.check_type('cell density', density, (Iterable, Real), none_ok=True)
        if isinstance(density, Iterable):
            cv.check_type('cell density', density, Iterable, Real)
            for rho in density:
                cv.check_greater_than('cell density', rho, 0.0, True)
        elif isinstance(density, Real):
            cv.check_greater_than('cell density', density, 0.0, True)
        if self.fill_type in ('universe', 'lattice'):
            for c in self.get_all_cells().values():
                if c.fill_type == 'material':
                    c._density = density
        else:
            self._density = density

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):
        cv.check_type('cell translation', translation, Iterable, Real)
        cv.check_length('cell translation', translation, 3)
        self._translation = np.asarray(translation, dtype=float)

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, volume):
        if volume is not None:
            cv.check_type('cell volume', volume, Real)
            cv.check_greater_than('cell volume', volume, 0.0, equality=True)
        self._volume = volume

    @property
    def bounding_box(self):
        if self.region is not None:
            return self.region.bounding_box
        return BoundingBox.infinite()

    def get_nuclides(self):
        """Returns all nuclides in the cell"""
        return self.fill.get_nuclides() if self.fill_type != 'void' else []

    def get_nuclide_densities(self):
        """Return all nuclides contained in the cell and their densities"""
        if self.fill_type == 'material':
            return dict(self.fill.get_nuclide_densities())
        if self.fill_type == 'void':
            return {}
        raise RuntimeError('Nuclide densities are only available for cells filled '
                           'with a material.')

    def get_all_cells(self, memo=None):
        """Return all cells that are contained within this one if it is filled
        with a universe or lattice"""
        if memo is None:
            memo = set()
        elif self in memo:
            return {}
        memo.add(self)
        cells = {}
        if self.fill_type in ('universe', 'lattice'):
            cells.update(self.fill.get_all_cells(memo))
        return cells

    def get_all_materials(self, memo=None):
        """Return all materials that are contained within the cell"""
        materials = {}
        if self.fill_type == 'material':
            materials[self.fill.id] = self.fill
        else:
            for cell in self.get_all_cells(memo).values():
                materials.update(cell.get_all_materials(memo))
        return materials

    def get_all_universes(self, memo=None):
        """Return all universes that are contained within this one if any of
        its cells are filled with a universe or lattice."""
        if memo is None:
            memo = set()
        if self in memo:
            return {}
        memo.add(self)
        universes = {}
        if self.fill_type == 'universe':
            universes[self.fill.id] = self.fill
            universes.update(self.fill.get_all_universes(memo))
        elif self.fill_type == 'lattice':
            universes.update(self.fill.get_all_universes(memo))
        return universes

    def clone(self, clone_materials=True, clone_regions=True, memo=None):
        """Create a copy of this cell with a new unique ID, and clones the
        cell's region and fill."""
        if memo is None:
            memo = {}
        if self not in memo:
            clone = Cell(name=self.name)
            clone.volume = self.volume
            if self.temperature is not None:
                clone.temperature = self.temperature
            if self.density is not None:
                clone.density = self.density
            if self.translation is not None:
                clone.translation = self.translation
            if self.rotation is not None:
                clone.rotation = self.rotation
            if self.region is not None:
                clone.region = self.region.clone(memo) if clone_regions else self.region
            if self.fill is not None:
                if self.fill_type == 'material':
                    clone.fill = self.fill.clone(memo) if clone_materials else self.fill
                else:
                    clone.fill = self.fill.clone(clone_materials, clone_regions, memo)
            memo[self] = clone
        return memo[self]

    def plot(self, *args, **kwargs):
        """Plotting is not supported by UM2; this is a no-op."""
        from ._stubs import _ignored
        _ignored('Cell.plot')

    def create_xml_subelement(self, xml_element, memo=None):
        """Add the cell's xml representation to an incoming xml element"""
        element = ET.Element("cell")
        element.set("id", str(self.id))
        if len(self._name) > 0:
            element.set("name", str(self.name))
        if self.fill_type == 'void':
            element.set("material", "void")
        elif self.fill_type == 'material':
            element.set("material", str(self.fill.id))
        else:
            element.set("fill", str(self.fill.id))
            self.fill.create_xml_subelement(xml_element, memo)

        if self.region is not None:
            region = str(self.region)
            if region.startswith('('):
                region = region[1:-1]
            if len(region) > 0:
                element.set("region", region)

            if memo is None:
                memo = set()

            def create_surface_elements(node):
                if isinstance(node, Halfspace):
                    if node.surface in memo:
                        return
                    memo.add(node.surface)
                    xml_element.append(node.surface.to_xml_element())
                elif isinstance(node, Complement):
                    create_surface_elements(node.node)
                else:
                    for subnode in node:
                        create_surface_elements(subnode)

            create_surface_elements(self.region)

        if self.temperature is not None:
            if isinstance(self.temperature, Iterable):
                sub = ET.SubElement(element, "temperature")
                sub.text = ' '.join(str(t) for t in self.temperature)
            else:
                element.set("temperature", str(self.temperature))
        if self.density is not None:
            if isinstance(self.density, Iterable):
                sub = ET.SubElement(element, "density")
                sub.text = ' '.join(str(d) for d in self.density)
            else:
                element.set("density", str(self.density))
        if self.translation is not None:
            element.set("translation", ' '.join(map(str, self.translation)))
        if self.rotation is not None:
            element.set("rotation", ' '.join(map(str, self.rotation.ravel())))
        if self.volume is not None:
            element.set("volume", str(self.volume))
        return element
