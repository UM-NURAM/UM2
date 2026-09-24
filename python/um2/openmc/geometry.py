"""Geometry (compatible with :class:`openmc.Geometry`)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from . import _xml as xml
from .checkvalue import check_greater_than, check_less_than, check_type
from .universe import Universe, UniverseBase


class Geometry:
    """Geometry representing a collection of surfaces, cells, and universes.

    Parameters
    ----------
    root : openmc.UniverseBase or Iterable of openmc.Cell, optional
        Root universe which contains all others, or an iterable of cells that
        should be used to create a root universe.
    merge_surfaces : bool
        Whether to remove redundant surfaces when the geometry is exported.
    surface_precision : int
        Number of decimal places to round to for comparing the coefficients of
        surfaces for considering them topologically equivalent.
    """

    def __init__(self, root: UniverseBase | Iterable | None = None,
                 merge_surfaces: bool = False, surface_precision: int = 10):
        self._root_universe = None
        self.merge_surfaces = merge_surfaces
        self.surface_precision = surface_precision
        if root is not None:
            if isinstance(root, UniverseBase):
                self.root_universe = root
            else:
                univ = Universe()
                for cell in root:
                    univ.add_cell(cell)
                self._root_universe = univ

    @property
    def root_universe(self) -> UniverseBase:
        return self._root_universe

    @root_universe.setter
    def root_universe(self, root_universe):
        check_type('root universe', root_universe, UniverseBase)
        self._root_universe = root_universe

    @property
    def bounding_box(self):
        return self.root_universe.bounding_box

    @property
    def merge_surfaces(self) -> bool:
        return self._merge_surfaces

    @merge_surfaces.setter
    def merge_surfaces(self, merge_surfaces):
        check_type('merge surfaces', merge_surfaces, bool)
        self._merge_surfaces = merge_surfaces

    @property
    def surface_precision(self) -> int:
        return self._surface_precision

    @surface_precision.setter
    def surface_precision(self, surface_precision):
        check_type('surface precision', surface_precision, int)
        check_less_than('surface_precision', surface_precision, 16)
        check_greater_than('surface_precision', surface_precision, 0)
        self._surface_precision = surface_precision

    def to_xml_element(self, remove_surfs=False) -> ET.Element:
        """Creates a 'geometry' element to be written to an XML file."""
        if remove_surfs:
            self.merge_surfaces = True
        if self.merge_surfaces:
            self.remove_redundant_surfaces()
        element = ET.Element("geometry")
        self.root_universe.create_xml_subelement(element)
        element[:] = sorted(element, key=lambda x: (x.tag, int(x.get('id'))))
        xml.clean_indentation(element)
        return element

    def export_to_xml(self, path='geometry.xml', remove_surfs=False):
        """Export geometry to an XML file.

        Parameters
        ----------
        path : str
            Path to file to write. Defaults to 'geometry.xml'.
        remove_surfs : bool
            Whether or not to remove redundant surfaces from the geometry when
            exporting
        """
        root_element = self.to_xml_element(remove_surfs)
        p = Path(path)
        if p.is_dir():
            p /= 'geometry.xml'
        ET.ElementTree(root_element).write(str(p), xml_declaration=True,
                                           encoding='utf-8')

    def find(self, point) -> list:
        """Find cells/universes/lattices which contain a given point"""
        return self.root_universe.find(point)

    def get_all_cells(self) -> dict:
        """Return all cells in the geometry."""
        if self.root_universe is not None:
            return self.root_universe.get_all_cells()
        return {}

    def get_all_universes(self) -> dict:
        """Return all universes in the geometry."""
        universes = {self.root_universe.id: self.root_universe}
        universes.update(self.root_universe.get_all_universes())
        return universes

    def get_all_nuclides(self) -> list[str]:
        """Return all nuclides within the geometry."""
        all_nuclides = set()
        for material in self.get_all_materials().values():
            all_nuclides |= set(material.get_nuclides())
        return sorted(all_nuclides)

    def get_all_materials(self) -> dict:
        """Return all materials within the geometry."""
        if self.root_universe is not None:
            return self.root_universe.get_all_materials()
        return {}

    def get_all_material_cells(self) -> dict:
        """Return all cells filled by a material"""
        return {cell.id: cell for cell in self.get_all_cells().values()
                if cell.fill_type == 'material'}

    def get_all_material_universes(self) -> dict:
        """Return all universes having at least one material-filled cell."""
        material_universes = {}
        for universe in self.get_all_universes().values():
            for cell in universe.cells.values():
                if cell.fill_type in ('material', 'void'):
                    material_universes[universe.id] = universe
        return material_universes

    def get_all_lattices(self) -> dict:
        """Return all lattices defined"""
        return {cell.fill.id: cell.fill for cell in self.get_all_cells().values()
                if cell.fill_type == 'lattice'}

    def get_all_surfaces(self) -> dict:
        """Return all surfaces used in the geometry"""
        surfaces = {}
        for cell in self.get_all_cells().values():
            if cell.region is not None:
                surfaces = cell.region.get_surfaces(surfaces)
        return surfaces

    def _get_domains_by_name(self, name, case_sensitive, matching, domain_type) -> list:
        if not case_sensitive:
            name = name.lower()
        domains = []
        func = getattr(self, f'get_all_{domain_type}s')
        for domain in func().values():
            domain_name = domain.name if case_sensitive else domain.name.lower()
            if domain_name == name:
                domains.append(domain)
            elif not matching and name in domain_name:
                domains.append(domain)
        domains.sort(key=lambda x: x.id)
        return domains

    def get_materials_by_name(self, name, case_sensitive=False, matching=False) -> list:
        """Return a list of materials with matching names."""
        return self._get_domains_by_name(name, case_sensitive, matching, 'material')

    def get_cells_by_name(self, name, case_sensitive=False, matching=False) -> list:
        """Return a list of cells with matching names."""
        return self._get_domains_by_name(name, case_sensitive, matching, 'cell')

    def get_surfaces_by_name(self, name, case_sensitive=False, matching=False) -> list:
        """Return a list of surfaces with matching names."""
        return self._get_domains_by_name(name, case_sensitive, matching, 'surface')

    def get_cells_by_fill_name(self, name, case_sensitive=False, matching=False) -> list:
        """Return a list of cells with fills with matching names."""
        if not case_sensitive:
            name = name.lower()
        cells = set()
        for cell in self.get_all_cells().values():
            if cell.fill_type == 'void':
                continue
            fill_name = cell.fill.name if case_sensitive else cell.fill.name.lower()
            if fill_name == name or (not matching and name in fill_name):
                cells.add(cell)
        return sorted(cells, key=lambda x: x.id)

    def get_universes_by_name(self, name, case_sensitive=False, matching=False) -> list:
        """Return a list of universes with matching names."""
        return self._get_domains_by_name(name, case_sensitive, matching, 'universe')

    def get_lattices_by_name(self, name, case_sensitive=False, matching=False) -> list:
        """Return a list of lattices with matching names."""
        return self._get_domains_by_name(name, case_sensitive, matching, 'lattice')

    def remove_redundant_surfaces(self) -> dict:
        """Remove and return all of the redundant surfaces."""
        redundancies = defaultdict(list)
        for surf in self.get_all_surfaces().values():
            coeffs = tuple(round(surf._coefficients[k], self.surface_precision)
                           for k in surf._coeff_keys)
            key = (surf._type, surf._boundary_type) + coeffs
            redundancies[key].append(surf)
        redundant_surfaces = {replace.id: keep
                              for keep, *redundant in redundancies.values()
                              for replace in redundant}
        if redundant_surfaces:
            for cell in self.get_all_cells().values():
                if cell.region:
                    cell.region.remove_redundant_surfaces(redundant_surfaces)
        return redundant_surfaces

    def clone(self) -> Geometry:
        """Create a copy of this geometry with new unique IDs for all of its
        enclosed materials, surfaces, cells, universes and lattices."""
        clone = deepcopy(self)
        clone.root_universe = self.root_universe.clone()
        return clone

    def plot(self, *args, **kwargs):
        """Plotting is not supported by UM2; this is a no-op."""
        from ._stubs import _ignored
        _ignored('Geometry.plot')
