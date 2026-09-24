"""Materials (compatible with :class:`openmc.Material` and
:class:`openmc.Materials`).

UM2 only uses macroscopic multigroup cross sections, so for meshing a material
must either reference a macroscopic data set with :meth:`Material.add_macroscopic`
or be meshed without cross sections. Nuclide and element compositions are still
fully supported so that OpenMC scripts can define (and export) them unchanged.
"""

from __future__ import annotations

from collections import Counter, defaultdict, namedtuple
from collections.abc import Iterable
from copy import deepcopy
from numbers import Real
from pathlib import Path
import re
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from . import checkvalue as cv
from . import data
from ._paths import input_path
from ._xml import clean_indentation, get_elem_list, get_text
from .checkvalue import PathLike
from .element import Element
from .mixin import IDManagerMixin

# Units for density supported by OpenMC
DENSITY_UNITS = ('g/cm3', 'g/cc', 'kg/m3', 'atom/b-cm', 'atom/cm3', 'sum',
                 'macro')

# Smallest normalized floating point number
_SMALLEST_NORMAL = sys.float_info.min

NuclideTuple = namedtuple('NuclideTuple', ['name', 'percent', 'percent_type'])


class Material(IDManagerMixin):
    """A material composed of a collection of nuclides/elements, or of a single
    macroscopic multigroup data set.

    Parameters
    ----------
    material_id : int, optional
        Unique identifier for the material. If not specified, an identifier will
        automatically be assigned.
    name : str, optional
        Name of the material. If not specified, the name will be the empty
        string.
    temperature : float, optional
        Temperature of the material in Kelvin.
    density : float, optional
        Density of the material (units defined separately)
    density_units : str
        Units used for `density`. Can be one of 'g/cm3', 'g/cc', 'kg/m3',
        'atom/b-cm', 'atom/cm3', 'sum', or 'macro'.
    depletable : bool, optional
        Indicate whether the material is depletable.
    volume : float, optional
        Volume of the material in cm^3.
    components : dict of str to float or dict
        Dictionary mapping element or nuclide names to their atom or weight
        percent.
    percent_type : {'ao', 'wo'}
        Whether the values in `components` are atom or weight percents.
    """

    next_id = 1
    used_ids = set()

    def __init__(
        self,
        material_id: int | None = None,
        name: str = "",
        temperature: float | None = None,
        density: float | None = None,
        density_units: str = "sum",
        depletable: bool | None = False,
        volume: float | None = None,
        components: dict | None = None,
        percent_type: str = "ao",
    ):
        self.id = material_id
        self.name = name
        self.temperature = temperature
        self._density = None
        self._density_units = density_units
        self._depletable = depletable
        self._volume = volume
        self._isotropic = []
        self._nuclides = []     # list of NuclideTuple
        self._macroscopic = None
        self._sab = []          # list of (name, fraction)

        if density is not None:
            self.set_density(density_units, density)
        if components is not None:
            self.add_components(components, percent_type=percent_type)

    def __repr__(self) -> str:
        string = 'Material\n'
        string += '{: <16}=\t{}\n'.format('\tID', self._id)
        string += '{: <16}=\t{}\n'.format('\tName', self._name)
        string += '{: <16}=\t{}\n'.format('\tTemperature', self._temperature)
        string += '{: <16}=\t{}'.format('\tDensity', self._density)
        string += f' [{self._density_units}]\n'
        string += '{: <16}=\t{} [cm^3]\n'.format('\tVolume', self._volume)
        string += '{: <16}=\t{}\n'.format('\tDepletable', self._depletable)
        string += '{: <16}\n'.format('\tS(a,b) Tables')
        for sab in self._sab:
            string += '{: <16}=\t{}\n'.format('\tS(a,b)', sab)
        string += '{: <16}\n'.format('\tNuclides')
        for nuclide, percent, percent_type in self._nuclides:
            string += '{: <16}'.format(f'\t{nuclide}')
            string += f'=\t{percent: <12} [{percent_type}]\n'
        if self._macroscopic is not None:
            string += '{: <16}\n'.format('\tMacroscopic Data')
            string += '{: <16}'.format(f'\t{self._macroscopic}')
        return string

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str | None):
        if name is not None:
            cv.check_type(f'name for Material ID="{self._id}"', name, str)
            self._name = name
        else:
            self._name = ''

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: Real | None):
        cv.check_type(f'Temperature for Material ID="{self._id}"',
                      temperature, (Real, type(None)))
        self._temperature = temperature

    @property
    def density(self) -> float | None:
        return self._density

    @property
    def density_units(self) -> str:
        return self._density_units

    @property
    def depletable(self) -> bool:
        return self._depletable

    @depletable.setter
    def depletable(self, depletable: bool):
        cv.check_type(f'Depletable flag for Material ID="{self._id}"',
                      depletable, bool)
        self._depletable = depletable

    @property
    def nuclides(self) -> list[NuclideTuple]:
        return self._nuclides

    @property
    def macroscopic(self) -> str | None:
        """Name of the macroscopic data set, or None."""
        return self._macroscopic

    @property
    def isotropic(self) -> list[str]:
        return self._isotropic

    @isotropic.setter
    def isotropic(self, isotropic: Iterable[str]):
        cv.check_iterable_type('Isotropic scattering nuclides', isotropic, str)
        self._isotropic = list(isotropic)

    @property
    def average_molar_mass(self) -> float:
        mass = 0.
        moles = 0.
        for nuc in self.nuclides:
            if nuc.percent_type == 'ao':
                mass += nuc.percent * data.atomic_mass(nuc.name)
                moles += nuc.percent
            else:
                moles += nuc.percent / data.atomic_mass(nuc.name)
                mass += nuc.percent
        if moles == 0.0:
            raise ValueError("Material has no nuclides; cannot compute molar mass")
        return mass / moles

    @property
    def volume(self) -> float | None:
        return self._volume

    @volume.setter
    def volume(self, volume: Real):
        if volume is not None:
            cv.check_type('material volume', volume, Real)
        self._volume = volume

    @property
    def fissionable_mass(self) -> float:
        if self.volume is None:
            raise ValueError("Volume must be set in order to determine mass.")
        density = 0.0
        for nuc, atoms_per_bcm in self.get_nuclide_atom_densities().items():
            if data.zam(nuc)[0] >= 90:
                density += 1e24 * atoms_per_bcm * data.atomic_mass(nuc) / data.AVOGADRO
        return density*self.volume

    # ==========================================================================
    # Composition
    # ==========================================================================

    def set_density(self, units: str, density: float | None = None):
        """Set the density of the material.

        Parameters
        ----------
        units : {'g/cm3', 'g/cc', 'kg/m3', 'atom/b-cm', 'atom/cm3', 'sum', 'macro'}
            Physical units of density.
        density : float, optional
            Value of the density. Must be specified unless units is given as
            'sum'.
        """
        cv.check_value('density units', units, DENSITY_UNITS)
        self._density_units = units
        if units == 'sum':
            if density is not None:
                warnings.warn(f'Density "{density}" for Material ID="{self.id}" '
                              'is ignored because the unit is "sum"')
        else:
            if density is None:
                raise ValueError(f'Unable to set the density for Material '
                                 f'ID="{self.id}" because a density value must be '
                                 'given when not using "sum" unit')
            cv.check_type(f'the density for Material ID="{self.id}"', density, Real)
            self._density = density

    def add_nuclide(self, nuclide: str, percent: float, percent_type: str = 'ao'):
        """Add a nuclide to the material.

        Parameters
        ----------
        nuclide : str
            Nuclide to add, e.g., 'Mo95'
        percent : float
            Atom or weight percent
        percent_type : {'ao', 'wo'}
            'ao' for atom percent and 'wo' for weight percent
        """
        cv.check_type('nuclide', nuclide, str)
        cv.check_type('percent', percent, Real)
        cv.check_value('percent type', percent_type, {'ao', 'wo'})
        cv.check_greater_than('percent', percent, 0, equality=True)
        if self._macroscopic is not None:
            raise ValueError(f'Unable to add a Nuclide to Material ID="{self._id}" '
                             'as a macroscopic data-set has already been added')
        try:
            Z, _, _ = data.zam(nuclide)
        except ValueError as e:
            warnings.warn(str(e))
        else:
            # For actinides, have the material be depletable by default
            if Z >= 89:
                self.depletable = True
        self._nuclides.append(NuclideTuple(nuclide, percent, percent_type))

    def add_components(self, components: dict, percent_type: str = 'ao'):
        """Add multiple elements or nuclides to a material.

        Parameters
        ----------
        components : dict of str to float or dict
            Dictionary mapping element or nuclide names to their atom or weight
            percent. To specify enrichment of an element, the entry of
            ``components`` for that element must instead be a dictionary
            containing the keyword arguments as well as a value for
            ``'percent'``.
        percent_type : {'ao', 'wo'}
            'ao' for atom percent and 'wo' for weight percent
        """
        for component, params in components.items():
            cv.check_type('component', component, str)
            if isinstance(params, Real):
                params = {'percent': params}
            else:
                cv.check_type('params', params, dict)
                if 'percent' not in params:
                    raise ValueError("An entry in the dictionary does not have "
                                     "a required key: 'percent'")
            params['percent_type'] = percent_type
            if not component.isalpha():
                self.add_nuclide(component, **params)
            else:
                self.add_element(component, **params)

    def remove_nuclide(self, nuclide: str):
        """Remove a nuclide from the material."""
        cv.check_type('nuclide', nuclide, str)
        for nuc in reversed(self.nuclides):
            if nuclide == nuc.name:
                self.nuclides.remove(nuc)

    def remove_element(self, element):
        """Remove an element from the material."""
        cv.check_type('element', element, str)
        for nuc in reversed(self.nuclides):
            if re.split(r'\d+', nuc.name)[0] == element:
                self.nuclides.remove(nuc)

    def add_macroscopic(self, macroscopic: str):
        """Add a macroscopic data set to the material. The name must match a
        data set (XSdata) in the multigroup cross section library.

        Parameters
        ----------
        macroscopic : str
            Macroscopic to add
        """
        if self._nuclides or self._sab:
            raise ValueError(f'Unable to add a Macroscopic data set to Material '
                             f'ID="{self._id}" with a macroscopic value '
                             f'"{macroscopic}" as an incompatible data member '
                             '(i.e., nuclide or S(a,b) table) has already been added')
        if not isinstance(macroscopic, str):
            raise ValueError(f'Unable to add a Macroscopic to Material '
                             f'ID="{self._id}" with a non-string value "{macroscopic}"')
        if self._macroscopic is not None:
            raise ValueError(f'Unable to add a Macroscopic to Material ID="{self._id}". '
                             'Only one Macroscopic allowed per Material.')
        self._macroscopic = macroscopic
        if self._density is None:
            self.set_density('macro', 1.0)

    def remove_macroscopic(self, macroscopic: str):
        """Remove a macroscopic from the material."""
        if not isinstance(macroscopic, str):
            raise ValueError(f'Unable to remove a Macroscopic "{macroscopic}" in '
                             f'Material ID="{self._id}" since it is not a string')
        if macroscopic == self._macroscopic:
            self._macroscopic = None

    def add_element(self, element: str, percent: float, percent_type: str = 'ao',
                    enrichment: float | None = None,
                    enrichment_target: str | None = None,
                    enrichment_type: str | None = None,
                    cross_sections: str | None = None):
        """Add a natural element to the material.

        Parameters
        ----------
        element : str
            Element to add, e.g., 'Zr' or 'Zirconium'
        percent : float
            Atom or weight percent
        percent_type : {'ao', 'wo'}, optional
            'ao' for atom percent and 'wo' for weight percent.
        enrichment : float, optional
            Enrichment of an enrichment_target nuclide in percent (ao or wo).
            If enrichment_target is not supplied then it is enrichment for U235
            in weight percent.
        enrichment_target : str, optional
            Single nuclide name to enrich from a natural composition
        enrichment_type : {'ao', 'wo'}, optional
            'ao' for enrichment as atom percent and 'wo' for weight percent.
        cross_sections : str, optional
            Location of cross_sections.xml file.
        """
        cv.check_type('nuclide', element, str)
        cv.check_type('percent', percent, Real)
        cv.check_greater_than('percent', percent, 0, equality=True)
        cv.check_value('percent type', percent_type, {'ao', 'wo'})

        if not element.isalpha():
            raise ValueError("Element name should be given by the "
                             "element's symbol or name, e.g., 'Zr', 'zirconium'")
        if len(element) > 2:
            el = element.lower()
            element = data.ELEMENT_SYMBOL.get(el)
            if element is None:
                raise ValueError(f'Element name "{el}" not recognised')
        else:
            if element[0].islower():
                raise ValueError(f'Element name "{element}" should start with an '
                                 'uppercase letter')
            if len(element) == 2 and element[1].isupper():
                raise ValueError(f'Element name "{element}" should end with a '
                                 'lowercase letter')
            if element not in list(data.ATOMIC_SYMBOL.values())[1:]:
                raise ValueError(f'Element name "{element}" not recognised')

        if self._macroscopic is not None:
            raise ValueError(f'Unable to add an Element to Material ID="{self._id}" '
                             'as a macroscopic data-set has already been added')

        if enrichment is not None and enrichment_target is None:
            if not isinstance(enrichment, Real):
                raise ValueError(f'Unable to add an Element to Material '
                                 f'ID="{self._id}" with a non-floating point '
                                 f'enrichment value "{enrichment}"')
            elif element != 'U':
                raise ValueError(f'Unable to use enrichment for element {element} '
                                 f'which is not uranium for Material ID="{self._id}"')
            cv.check_less_than('enrichment', enrichment, 100./1.008)
            cv.check_greater_than('enrichment', enrichment, 0., equality=True)
            if enrichment > 5.0:
                warnings.warn(f'A uranium enrichment of {enrichment} was given for '
                              f'Material ID="{self._id}". OpenMC assumes the '
                              'U234/U235 mass ratio is constant at 0.008, which is '
                              'only valid at low enrichments. Consider setting the '
                              'isotopic composition manually for enrichments over 5%.')

        for nuclide in Element(element).expand(percent, percent_type, enrichment,
                                               enrichment_target, enrichment_type,
                                               cross_sections):
            self.add_nuclide(*nuclide)

    def add_elements_from_formula(self, formula: str, percent_type: str = 'ao',
                                  enrichment: float | None = None,
                                  enrichment_target: str | None = None,
                                  enrichment_type: str | None = None):
        """Add a elements from a chemical formula to the material.

        Parameters
        ----------
        formula : str
            Formula to add, e.g., 'C2O', 'C6H12O6', or (NH4)2SO4.
        percent_type, enrichment, enrichment_target, enrichment_type
            See :meth:`Material.add_element`
        """
        cv.check_type('formula', formula, str)
        if '.' in formula:
            raise ValueError('Non-integer multiplier values are not accepted. The '
                             f'input formula {formula} contains a "." character.')

        tokens = re.findall(r"([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)", formula)
        for row in tokens:
            for token in row:
                if token.isalpha():
                    if token == "n" or token not in data.ATOMIC_NUMBER:
                        raise ValueError(f'Formula entry {token} not an element '
                                         'symbol.')
                elif token not in ['(', ')', ''] and not token.isdigit():
                    raise ValueError('Formula must be made from a sequence of '
                                     'element symbols, integers, and brackets. '
                                     f'{token} is not an allowable entry.')
        if formula.count('(') != formula.count(')'):
            raise ValueError('Number of opening and closing brackets is not equal '
                             f'in the input formula {formula}.')
        remainder = formula
        for row in tokens:
            for token in row:
                remainder = remainder.replace(token, '', 1)
        if len(remainder) != 0:
            raise ValueError('Part of formula was not successfully parsed as an '
                             'element symbol, bracket or integer. '
                             f'{remainder} was not parsed.')

        mat_stack = [Counter()]
        for symbol, multi1, opening_bracket, closing_bracket, multi2 in tokens:
            if symbol:
                mat_stack[-1][symbol] += int(multi1 or 1)
            if opening_bracket:
                mat_stack.append(Counter())
            if closing_bracket:
                stack_top = mat_stack.pop()
                for sym, value in stack_top.items():
                    mat_stack[-1][sym] += int(multi2 or 1) * value

        percents = mat_stack[0].values()
        norm_percents = [float(i) / sum(percents) for i in percents]
        for element, percent in zip(mat_stack[0].keys(), norm_percents):
            if (enrichment_target is not None
                    and element == re.sub(r'\d+$', '', enrichment_target)):
                self.add_element(element, percent, percent_type, enrichment,
                                 enrichment_target, enrichment_type)
            elif enrichment is not None and enrichment_target is None and element == 'U':
                self.add_element(element, percent, percent_type, enrichment)
            else:
                self.add_element(element, percent, percent_type)

    def add_s_alpha_beta(self, name: str, fraction: float = 1.0):
        r"""Add an :math:`S(\alpha,\beta)` table to the material.

        The table is stored (and exported) for compatibility with OpenMC, but it
        is not used by UM2.
        """
        if self._macroscopic is not None:
            raise ValueError(f'Unable to add an S(a,b) table to Material '
                             f'ID="{self._id}" as a macroscopic data-set has '
                             'already been added')
        if not isinstance(name, str):
            raise ValueError(f'Unable to add an S(a,b) table to Material '
                             f'ID="{self._id}" with a non-string table name "{name}"')
        cv.check_type('S(a,b) fraction', fraction, Real)
        cv.check_greater_than('S(a,b) fraction', fraction, 0.0, True)
        cv.check_less_than('S(a,b) fraction', fraction, 1.0, True)
        self._sab.append((name, fraction))

    def make_isotropic_in_lab(self):
        self.isotropic = [x.name for x in self._nuclides]

    # ==========================================================================
    # Queries
    # ==========================================================================

    def get_elements(self) -> list[str]:
        """Returns all elements in the material"""
        return sorted({re.split(r'(\d+)', i)[0] for i in self.get_nuclides()})

    def get_nuclides(self, element: str | None = None) -> list[str]:
        """Returns a list of all nuclides in the material, if the element
        argument is specified then just nuclides of that element are returned.
        """
        matching = []
        for nuclide in self._nuclides:
            if element and re.split(r'(\d+)', nuclide.name)[0] != element:
                continue
            if nuclide.name not in matching:
                matching.append(nuclide.name)
        return matching

    def get_nuclide_densities(self) -> dict[str, tuple]:
        """Returns all nuclides in the material and their densities"""
        return {nuclide.name: nuclide for nuclide in self._nuclides}

    def get_nuclide_atom_densities(self, nuclide: str | None = None) -> dict[str, float]:
        """Returns one or all nuclides in the material and their atomic
        densities in units of atom/b-cm"""
        sum_density = False
        units = self.density_units
        if units == 'sum':
            sum_density = True
            density = 0.
        elif units == 'macro':
            density = self.density
        elif units in ('g/cc', 'g/cm3'):
            density = -self.density
        elif units == 'kg/m3':
            density = -0.001 * self.density
        elif units == 'atom/b-cm':
            density = self.density
        elif units in ('atom/cm3', 'atom/cc'):
            density = 1.e-24 * self.density

        nucs = [nuc.name for nuc in self.nuclides]
        nuc_densities = np.array([nuc.percent for nuc in self.nuclides], dtype=float)
        nuc_density_types = np.array([nuc.percent_type for nuc in self.nuclides])

        if sum_density:
            density = np.sum(nuc_densities)

        percent_in_atom = np.all(nuc_density_types == 'ao')
        density_in_atom = density > 0.

        if not percent_in_atom:
            amm = self.average_molar_mass
            for n, nuc in enumerate(nucs):
                nuc_densities[n] *= amm / data.atomic_mass(nuc)

        nuc_densities = nuc_densities / np.sum(nuc_densities)

        if not density_in_atom:
            density = -density / self.average_molar_mass * 1.e-24 * data.AVOGADRO

        nuc_densities = density * nuc_densities
        return {nuc: nuc_densities[n] for n, nuc in enumerate(nucs)
                if nuclide is None or nuclide == nuc}

    def get_element_atom_densities(self, element: str | None = None) -> dict[str, float]:
        """Returns one or all elements in the material and their atomic
        densities in units of atom/b-cm"""
        if element is not None:
            element = data.data._get_element_symbol(element)
        densities = {}
        for nuclide, density in self.get_nuclide_atom_densities().items():
            nuc_element = data.ATOMIC_SYMBOL[data.zam(nuclide)[0]]
            if element is None or element == nuc_element:
                densities[nuc_element] = densities.get(nuc_element, 0.0) + float(density)
        if element is not None and element not in densities:
            raise ValueError(f'Element {element} not found in material.')
        return densities

    def get_nuclide_atoms(self, volume: float | None = None) -> dict[str, float]:
        """Return number of atoms of each nuclide in the material"""
        if volume is None:
            volume = self.volume
        if volume is None:
            raise ValueError("Volume must be set in order to determine atoms.")
        return {nuclide: 1.0e24 * atom_per_bcm * volume
                for nuclide, atom_per_bcm in self.get_nuclide_atom_densities().items()}

    def get_mass_density(self, nuclide: str | None = None) -> float:
        """Return mass density of one or all nuclides in [g/cm^3]"""
        mass_density = 0.0
        for nuc, atoms_per_bcm in self.get_nuclide_atom_densities(nuclide=nuclide).items():
            mass_density += 1e24 * atoms_per_bcm * data.atomic_mass(nuc) / data.AVOGADRO
        return mass_density

    def get_mass(self, nuclide: str | None = None, volume: float | None = None) -> float:
        """Return mass of one or all nuclides in [g]"""
        if volume is None:
            volume = self.volume
        if volume is None:
            raise ValueError("Volume must be set in order to determine mass.")
        return volume*self.get_mass_density(nuclide)

    def clone(self, memo: dict | None = None) -> Material:
        """Create a copy of this material with a new unique ID."""
        if memo is None:
            memo = {}
        if self not in memo:
            clone = deepcopy(self)
            clone.id = None
            memo[self] = clone
        return memo[self]

    # ==========================================================================
    # XML
    # ==========================================================================

    def _get_nuclide_xml(self, nuclide: NuclideTuple) -> ET.Element:
        xml_element = ET.Element("nuclide")
        xml_element.set("name", nuclide.name)
        val = nuclide.percent
        if abs(val) < _SMALLEST_NORMAL:
            val = 0.0
        xml_element.set(nuclide.percent_type, str(val))
        return xml_element

    def to_xml_element(self, nuclides_to_ignore: Iterable[str] | None = None) -> ET.Element:
        """Return XML representation of the material"""
        element = ET.Element("material")
        element.set("id", str(self._id))
        if len(self._name) > 0:
            element.set("name", str(self._name))
        if self._depletable:
            element.set("depletable", "true")
        if self._volume:
            element.set("volume", str(self._volume))
        if self.temperature is not None:
            element.set("temperature", str(self.temperature))

        if self._density is not None or self._density_units == 'sum':
            subelement = ET.SubElement(element, "density")
            if self._density_units != 'sum':
                subelement.set("value", str(self._density))
            subelement.set("units", self._density_units)
        else:
            raise ValueError(f'Density has not been set for material {self.id}!')

        if self._macroscopic is None:
            for nuclide in self._nuclides:
                if nuclides_to_ignore and nuclide.name in nuclides_to_ignore:
                    continue
                element.append(self._get_nuclide_xml(nuclide))
        else:
            subelement = ET.SubElement(element, "macroscopic")
            subelement.set("name", self._macroscopic)

        for name, fraction in self._sab:
            subelement = ET.SubElement(element, "sab")
            subelement.set("name", name)
            if fraction != 1.0:
                subelement.set("fraction", str(fraction))

        if self._isotropic:
            subelement = ET.SubElement(element, "isotropic")
            subelement.text = ' '.join(self._isotropic)
        return element

    @classmethod
    def mix_materials(cls, materials, fracs: Iterable[float],
                      percent_type: str = 'ao', **kwargs) -> Material:
        """Mix materials together based on atom, weight, or volume fractions

        Parameters
        ----------
        materials : Iterable of openmc.Material
            Materials to combine
        fracs : Iterable of float
            Fractions of each material to be combined
        percent_type : {'ao', 'wo', 'vo'}
            Type of percentage, must be one of 'ao', 'wo', or 'vo', to signify
            atom percent (molar percent), weight percent, or volume percent,
            optional. Defaults to 'ao'
        **kwargs
            Keyword arguments passed to :class:`openmc.Material`
        """
        cv.check_type('materials', materials, Iterable, Material)
        cv.check_type('fracs', fracs, Iterable, Real)
        cv.check_value('percent type', percent_type, {'ao', 'wo', 'vo'})

        fracs = np.asarray(fracs)
        void_frac = 1. - np.sum(fracs)
        if not np.isclose(void_frac, 0.):
            if percent_type in ('ao', 'wo'):
                raise ValueError('A non-zero void fraction is not acceptable for '
                                 f'percent_type: {percent_type}')
            warnings.warn('Warning: sum of fractions do not add to 1, void '
                          f'fraction set to {void_frac}')

        amms = np.asarray([mat.average_molar_mass for mat in materials])
        mass_dens = np.asarray([mat.get_mass_density() for mat in materials])
        if percent_type == 'ao':
            wgts = fracs * amms / mass_dens
            wgts /= np.sum(wgts)
        elif percent_type == 'wo':
            wgts = fracs / mass_dens
            wgts /= np.sum(wgts)
        else:
            wgts = fracs

        if any(mat._sab for mat in materials):
            raise NotImplementedError('Currently we do not support mixing materials '
                                      'containing S(a,b) tables')

        nuclides_per_cc = defaultdict(float)
        mass_per_cc = defaultdict(float)
        for mat, wgt in zip(materials, wgts):
            for nuc, atoms_per_bcm in mat.get_nuclide_atom_densities().items():
                nuc_per_cc = wgt*1.e24*atoms_per_bcm
                nuclides_per_cc[nuc] += nuc_per_cc
                mass_per_cc[nuc] += nuc_per_cc*data.atomic_mass(nuc) / data.AVOGADRO

        if "name" not in kwargs:
            kwargs["name"] = '-'.join([f'{m.name}({f})' for m, f in zip(materials, fracs)])
        new_mat = cls(**kwargs)
        tot_nuclides_per_cc = np.sum(list(nuclides_per_cc.values()))
        for nuc, atom_dens in nuclides_per_cc.items():
            new_mat.add_nuclide(nuc, atom_dens/tot_nuclides_per_cc, 'ao')
        new_mat.set_density('g/cm3', np.sum(list(mass_per_cc.values())))
        new_mat.depletable = any(mat.depletable for mat in materials)
        return new_mat

    @classmethod
    def from_xml_element(cls, elem: ET.Element) -> Material:
        """Generate material from an XML element"""
        mat = cls(int(get_text(elem, 'id')))
        mat.name = get_text(elem, 'name')
        temperature = get_text(elem, "temperature")
        if temperature is not None:
            mat.temperature = float(temperature)
        volume = get_text(elem, "volume")
        if volume is not None:
            mat.volume = float(volume)
        for nuclide in elem.findall('nuclide'):
            name = get_text(nuclide, "name")
            if 'ao' in nuclide.attrib:
                mat.add_nuclide(name, float(nuclide.attrib['ao']))
            elif 'wo' in nuclide.attrib:
                mat.add_nuclide(name, float(nuclide.attrib['wo']), 'wo')
        macroscopic = elem.find('macroscopic')
        if macroscopic is not None:
            mat.add_macroscopic(get_text(macroscopic, 'name'))
        mat.depletable = get_text(elem, "depletable") in ('true', '1')
        for sab in elem.findall('sab'):
            mat.add_s_alpha_beta(get_text(sab, "name"),
                                 float(get_text(sab, "fraction", 1.0)))
        density = elem.find('density')
        units = get_text(density, "units")
        if units == 'sum':
            mat.set_density(units)
        else:
            mat.set_density(units, float(get_text(density, 'value')))
        isotropic = get_elem_list(elem, "isotropic", str)
        if isotropic is not None:
            mat.isotropic = isotropic
        return mat


class Materials(cv.CheckedList):
    """Collection of Materials used for an OpenMC simulation.

    Parameters
    ----------
    materials : Iterable of openmc.Material
        Materials to add to the collection

    Attributes
    ----------
    cross_sections : str or path-like
        Indicates the path to a cross section library. For UM2, this is the
        multigroup library (OpenMC MGXS HDF5 or MPACT .xsl) used by materials
        that call :meth:`Material.add_macroscopic`.
    """

    def __init__(self, materials=None):
        super().__init__(Material, 'materials collection')
        self._cross_sections = None
        if materials is not None:
            self += materials

    @property
    def cross_sections(self) -> Path | None:
        return self._cross_sections

    @cross_sections.setter
    def cross_sections(self, cross_sections):
        if cross_sections is not None:
            self._cross_sections = input_path(cross_sections)

    def append(self, material):
        """Append material to collection"""
        super().append(material)

    def insert(self, index: int, material):
        """Insert material before index"""
        super().insert(index, material)

    def make_isotropic_in_lab(self):
        for material in self:
            material.make_isotropic_in_lab()

    def _write_xml(self, file, header=True, level=0, spaces_per_level=2,
                   trailing_indent=True, nuclides_to_ignore=None):
        indentation = level*spaces_per_level*' '
        if header:
            file.write("<?xml version='1.0' encoding='utf-8'?>\n")
        file.write(indentation + '<materials>\n')
        if self.cross_sections is not None:
            element = ET.Element('cross_sections')
            element.text = str(self.cross_sections)
            clean_indentation(element, level=level+1)
            element.tail = element.tail.strip(' ')
            file.write((level+1)*spaces_per_level*' ')
            file.write(ET.tostring(element, encoding="unicode"))
        for material in sorted(set(self), key=lambda x: x.id):
            element = material.to_xml_element(nuclides_to_ignore=nuclides_to_ignore)
            clean_indentation(element, level=level+1)
            element.tail = element.tail.strip(' ')
            file.write((level+1)*spaces_per_level*' ')
            file.write(ET.tostring(element, encoding="unicode"))
        file.write(indentation + '</materials>\n')
        if trailing_indent:
            file.write(indentation)

    def export_to_xml(self, path: PathLike = 'materials.xml',
                      nuclides_to_ignore: Iterable[str] | None = None):
        """Export material collection to an XML file.

        Parameters
        ----------
        path : str
            Path to file to write. Defaults to 'materials.xml'.
        nuclides_to_ignore : list of str
            Nuclides to ignore when exporting to XML.
        """
        p = Path(path)
        if p.is_dir():
            p /= 'materials.xml'
        with open(str(p), 'w', encoding='utf-8', errors='xmlcharrefreplace') as fh:
            self._write_xml(fh, nuclides_to_ignore=nuclides_to_ignore)

    @classmethod
    def from_xml_element(cls, elem) -> Materials:
        """Generate materials collection from XML file"""
        materials = cls()
        for material in elem.findall('material'):
            materials.append(Material.from_xml_element(material))
        xs = get_text(elem, "cross_sections")
        if xs is not None:
            materials.cross_sections = xs
        return materials

    @classmethod
    def from_xml(cls, path: PathLike = 'materials.xml') -> Materials:
        """Generate materials collection from XML file"""
        return cls.from_xml_element(ET.parse(path).getroot())
