"""Universes (compatible with :class:`openmc.Universe`)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Real

import numpy as np

from ..errors import UM2NotSupportedError
from . import checkvalue as cv
from .bounding_box import BoundingBox
from .cell import Cell
from .mixin import IDManagerMixin
from .region import Union


class UniverseBase(ABC, IDManagerMixin):
    """A collection of cells that can be repeated.

    Parameters
    ----------
    universe_id : int, optional
        Unique identifier of the universe. If not specified, an identifier will
        automatically be assigned
    name : str, optional
        Name of the universe. If not specified, the name is the empty string.
    """

    next_id = 1
    used_ids = set()

    def __init__(self, universe_id=None, name=''):
        self.id = universe_id
        self.name = name
        self._volume = None
        self._cells = {}

    def __repr__(self):
        string = 'Universe\n'
        string += '{: <16}=\t{}\n'.format('\tID', self._id)
        string += '{: <16}=\t{}\n'.format('\tName', self._name)
        return string

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            cv.check_type('universe name', name, str)
            self._name = name
        else:
            self._name = ''

    @property
    def cells(self):
        return self._cells

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, volume):
        if volume is not None:
            cv.check_type('universe volume', volume, Real)
        self._volume = volume

    def get_all_universes(self, memo=None):
        """Return all universes that are contained within this one."""
        if memo is None:
            memo = set()
        elif self in memo:
            return {}
        memo.add(self)
        universes = {}
        for cell in self.get_all_cells().values():
            universes.update(cell.get_all_universes(memo))
        return universes

    @abstractmethod
    def create_xml_subelement(self, xml_element, memo=None):
        """Add the universe xml representation to an incoming xml element"""

    def add_cells(self, cells):
        """Add multiple cells to the universe.

        Parameters
        ----------
        cells : Iterable of openmc.Cell
            Cells to add
        """
        if not isinstance(cells, Iterable):
            raise TypeError(f'Unable to add Cells to Universe ID="{self._id}" since '
                            f'"{cells}" is not iterable')
        for cell in cells:
            self.add_cell(cell)

    @abstractmethod
    def add_cell(self, cell):
        pass

    @abstractmethod
    def remove_cell(self, cell):
        pass

    def clear_cells(self):
        """Remove all cells from the universe."""
        self._cells.clear()

    def get_all_cells(self, memo=None):
        """Return all cells that are contained within the universe"""
        if memo is None:
            memo = set()
        elif self in memo:
            return {}
        memo.add(self)
        cells = dict(self._cells)
        for cell in self._cells.values():
            cells.update(cell.get_all_cells(memo))
        return cells

    def get_all_materials(self, memo=None):
        """Return all materials that are contained within the universe"""
        if memo is None:
            memo = set()
        materials = {}
        for cell in self.get_all_cells(memo).values():
            materials.update(cell.get_all_materials(memo))
        return materials

    @abstractmethod
    def _partial_deepcopy(self):
        """Deepcopy all parameters of an openmc.UniverseBase object except its
        cells."""

    def clone(self, clone_materials=True, clone_regions=True, memo=None):
        """Create a copy of this universe with a new unique ID, and clones all
        cells within this universe."""
        if memo is None:
            memo = {}
        if self not in memo:
            clone = self._partial_deepcopy()
            clone._cells = {}
            for cell in self._cells.values():
                clone.add_cell(cell.clone(clone_materials, clone_regions, memo))
            memo[self] = clone
        return memo[self]

    def find(self, point):
        """Find cells/universes/lattices which contain a given point

        Parameters
        ----------
        point : 3-tuple of float
            Cartesian coordinates of the point

        Returns
        -------
        list
            Sequence of universes, cells, and lattices which are traversed to
            find the given point
        """
        p = np.asarray(point, dtype=float)
        for cell in self._cells.values():
            if p in cell:
                if cell.fill_type in ('material', 'void'):
                    return [self, cell]
                elif cell.fill_type == 'universe':
                    if cell.translation is not None:
                        p = p - cell.translation
                    if cell.rotation is not None:
                        p = cell.rotation_matrix.dot(p)
                    return [self, cell] + cell.fill.find(p)
                else:
                    return [self, cell] + cell.fill.find(p)
        return []

    def plot(self, *args, **kwargs):
        """Plotting is not supported by UM2; this is a no-op.

        The geometry can be visualized with gmsh after it has been built with
        :func:`um2.build_geometry` (e.g. ``gmsh.fltk.run()``).
        """
        from ._stubs import _ignored
        _ignored('Universe.plot')

    def get_nuclides(self):
        """Returns all nuclides in the universe"""
        nuclides = []
        for cell in self.cells.values():
            for nuclide in cell.get_nuclides():
                if nuclide not in nuclides:
                    nuclides.append(nuclide)
        return nuclides


class Universe(UniverseBase):
    """A collection of cells that can be repeated.

    Parameters
    ----------
    universe_id : int, optional
        Unique identifier of the universe. If not specified, an identifier will
        automatically be assigned
    name : str, optional
        Name of the universe. If not specified, the name is the empty string.
    cells : Iterable of openmc.Cell, optional
        Cells to add to the universe. By default no cells are added.
    """

    def __init__(self, universe_id=None, name='', cells=None):
        super().__init__(universe_id, name)
        if cells is not None:
            self.add_cells(cells)

    def __repr__(self):
        string = super().__repr__()
        string += '{: <16}=\t{}\n'.format('\tGeom', 'CSG')
        string += '{: <16}=\t{}\n'.format('\tCells', list(self._cells.keys()))
        return string

    @property
    def bounding_box(self) -> BoundingBox:
        regions = [c.region for c in self.cells.values() if c.region is not None]
        if regions:
            return Union(regions).bounding_box
        return BoundingBox.infinite()

    def add_cell(self, cell):
        """Add a cell to the universe.

        Parameters
        ----------
        cell : openmc.Cell
            Cell to add
        """
        if not isinstance(cell, Cell):
            raise TypeError(f'Unable to add a Cell to Universe ID="{self._id}" since '
                            f'"{cell}" is not a Cell')
        if cell.id not in self._cells:
            self._cells[cell.id] = cell

    def remove_cell(self, cell):
        """Remove a cell from the universe.

        Parameters
        ----------
        cell : openmc.Cell
            Cell to remove
        """
        if not isinstance(cell, Cell):
            raise TypeError(f'Unable to remove a Cell from Universe ID="{self._id}" '
                            f'since "{cell}" is not a Cell')
        self._cells.pop(cell.id, None)

    def create_xml_subelement(self, xml_element, memo=None):
        if memo is None:
            memo = set()
        for cell in self._cells.values():
            if cell in memo:
                continue
            memo.add(cell)
            cell_element = cell.create_xml_subelement(xml_element, memo)
            cell_element.set("universe", str(self._id))
            xml_element.append(cell_element)

    def _partial_deepcopy(self):
        clone = Universe(name=self.name)
        clone.volume = self.volume
        return clone


class DAGMCUniverse(UniverseBase):
    """Not supported by UM2 (CAD-based DAGMC geometry). Creating one raises
    :class:`um2.errors.UM2NotSupportedError`."""

    _um2_unsupported = True

    def __init__(self, *args, **kwargs):
        raise UM2NotSupportedError('openmc.DAGMCUniverse is not supported by UM2.')

    def add_cell(self, cell):
        pass

    def remove_cell(self, cell):
        pass

    def create_xml_subelement(self, xml_element, memo=None):
        pass

    def _partial_deepcopy(self):
        pass
