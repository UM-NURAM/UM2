"""Natural elements (compatible with :class:`openmc.Element`)."""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET

from . import checkvalue as cv
from .data import NATURAL_ABUNDANCE, atomic_mass, isotopes as natural_isotopes, zam


def _to_mass_fractions(abundances):
    """Convert atom fractions to normalized mass fractions (in place)."""
    element_am = sum(atomic_mass(nuc) * frac for nuc, frac in abundances.items())
    for nuc in abundances:
        abundances[nuc] *= atomic_mass(nuc) / element_am
    _normalize(abundances)


def _to_atom_fractions(abundances):
    """Convert mass fractions to normalized atom fractions (in place)."""
    for nuc in abundances:
        abundances[nuc] /= atomic_mass(nuc)
    _normalize(abundances)


def _normalize(abundances):
    total = sum(abundances.values())
    for nuc in abundances:
        abundances[nuc] /= total


class Element(str):
    """A natural element that auto-expands to add the isotopes of an element to
    a material in their natural abundance.

    Parameters
    ----------
    name : str
        Chemical symbol of the element, e.g. Pu
    """

    def __new__(cls, name):
        cv.check_type('element name', name, str)
        cv.check_length('element name', name, 1, 2)
        return super().__new__(cls, name)

    @property
    def name(self):
        return self

    def expand(self, percent, percent_type, enrichment=None,
               enrichment_target=None, enrichment_type=None,
               cross_sections=None):
        """Expand natural element into its naturally-occurring isotopes.

        See :meth:`openmc.Element.expand` for a description of the parameters.

        Returns
        -------
        isotopes : list
            Naturally-occurring isotopes of the element. Each item of the list
            is a tuple consisting of a nuclide string, the atom/weight percent,
            and the string 'ao' or 'wo'.
        """
        if enrichment_type is not None:
            cv.check_value('enrichment_type', enrichment_type, {'ao', 'wo'})
        if enrichment is not None:
            cv.check_less_than('enrichment', enrichment, 100.0, equality=True)
            cv.check_greater_than('enrichment', enrichment, 0., equality=True)

        natural_nuclides = {name for name, _ in natural_isotopes(self)}
        if len(natural_nuclides) == 0:
            warnings.warn(f"No naturally occurring isotopes found for {self}.")

        if cross_sections is None:
            from .config import config
            cross_sections = config.get('cross_sections')

        abundances = {}
        if cross_sections is not None:
            abundances = self._library_abundances(natural_nuclides, cross_sections)
        else:
            for nuclide in sorted(natural_nuclides, key=zam):
                abundances[nuclide] = NATURAL_ABUNDANCE[nuclide]

        if enrichment is not None and enrichment_target is None:
            # Uranium enrichment (weight percent U235)
            if self.name != 'U':
                raise ValueError('Enrichment procedure for Uranium was requested, '
                                 f'but the isotope is {self} not U')
            if enrichment_type == 'ao':
                raise ValueError('Enrichment procedure for Uranium requires that '
                                 'enrichment value is provided as wo%.')
            abundances['U234'] = 0.0089 * enrichment
            abundances['U235'] = enrichment
            abundances['U236'] = 0.0046 * enrichment
            abundances['U238'] = 100.0 - 1.0135 * enrichment
            _to_atom_fractions(abundances)

        elif enrichment is not None and enrichment_target is not None:
            # Enrichment of a two-isotope element
            if enrichment_target == 'U235':
                raise ValueError("There is a special procedure for enrichment of "
                                 "U235 in U. To invoke it, the arguments "
                                 "'enrichment_target' and 'enrichment_type' "
                                 "should be omitted. Provide a value only for "
                                 "'enrichment' in weight percent.")
            if len(abundances) != 2:
                raise ValueError(f'Element {self} does not consist of two '
                                 'naturally-occurring isotopes. Please enter '
                                 'isotopic abundances manually.')
            if enrichment_target not in abundances:
                raise ValueError(f'The target nuclide {enrichment_target} is not '
                                 'one of the naturally-occurring isotopes '
                                 f'({list(abundances)})')
            if enrichment_type == 'wo':
                _to_mass_fractions(abundances)
            non_enriched = 1.0 - abundances[enrichment_target]
            tail_fraction = 1.0 - enrichment / 100.0
            for nuclide, fraction in abundances.items():
                abundances[nuclide] = tail_fraction * fraction / non_enriched
            abundances[enrichment_target] = enrichment / 100.0
            if enrichment_type == 'wo':
                _to_atom_fractions(abundances)

        if percent_type == 'wo':
            _to_mass_fractions(abundances)

        return [(nuclide, percent * abundance, percent_type)
                for nuclide, abundance in abundances.items()]

    def _library_abundances(self, natural_nuclides, cross_sections):
        """Restrict natural nuclides to those present in a cross_sections.xml"""
        library_nuclides = set()
        root = ET.parse(cross_sections).getroot()
        for child in root.findall('library'):
            nuclide = child.attrib['materials']
            if re.match(rf'{self}\d+', nuclide):
                library_nuclides.add(nuclide)

        mutual = sorted(natural_nuclides & library_nuclides, key=zam)
        absent = sorted(natural_nuclides - library_nuclides, key=zam)
        abundances = {}
        if not absent:
            for nuclide in mutual:
                abundances[nuclide] = NATURAL_ABUNDANCE[nuclide]
        elif (self + '0') in library_nuclides:
            abundances[self + '0'] = 1.0
        elif not mutual:
            raise ValueError(f'Unable to expand element {self} because the cross '
                             'section library provided does not contain any of '
                             'the natural isotopes for that element.')
        else:
            for nuclide in mutual:
                abundances[nuclide] = NATURAL_ABUNDANCE[nuclide]
            for nuclide in absent:
                if nuclide in ['O17', 'O18'] and 'O16' in mutual:
                    abundances['O16'] += NATURAL_ABUNDANCE[nuclide]
                elif nuclide == 'Ta180_m1' and 'Ta181' in mutual:
                    abundances['Ta181'] += NATURAL_ABUNDANCE[nuclide]
                elif nuclide == 'W180' and 'W182' in mutual:
                    abundances['W182'] += NATURAL_ABUNDANCE[nuclide]
                else:
                    raise ValueError(
                        f'Unsure how to partition natural abundance of isotope '
                        f'{nuclide} into other natural isotopes of this element '
                        'that are present in the cross section library provided. '
                        'Consider adding the isotopes of this element individually.')
        return abundances
