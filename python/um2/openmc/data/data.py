"""Nuclear data constants and helpers (subset of :mod:`openmc.data`).

Natural abundances and atomic masses are generated from OpenMC by
python/tools/sync_openmc.py. Element names and symbols replace the tables that
OpenMC imports from the ``endf`` package.
"""

from __future__ import annotations

import itertools
import os
import re

from ._natural_abundance import NATURAL_ABUNDANCE

# Physical constants (same values as OpenMC)
JOULE_PER_EV = 1.602176634e-19
AVOGADRO = 6.02214076e23
NEUTRON_MASS = 1.00866491595
K_BOLTZMANN = 8.617333262e-5  # eV/K
EV_PER_MEV = 1.0e6

_ELEMENTS = (
    'hydrogen', 'helium', 'lithium', 'beryllium', 'boron', 'carbon', 'nitrogen',
    'oxygen', 'fluorine', 'neon', 'sodium', 'magnesium', 'aluminum', 'silicon',
    'phosphorus', 'sulfur', 'chlorine', 'argon', 'potassium', 'calcium',
    'scandium', 'titanium', 'vanadium', 'chromium', 'manganese', 'iron',
    'cobalt', 'nickel', 'copper', 'zinc', 'gallium', 'germanium', 'arsenic',
    'selenium', 'bromine', 'krypton', 'rubidium', 'strontium', 'yttrium',
    'zirconium', 'niobium', 'molybdenum', 'technetium', 'ruthenium', 'rhodium',
    'palladium', 'silver', 'cadmium', 'indium', 'tin', 'antimony', 'tellurium',
    'iodine', 'xenon', 'cesium', 'barium', 'lanthanum', 'cerium',
    'praseodymium', 'neodymium', 'promethium', 'samarium', 'europium',
    'gadolinium', 'terbium', 'dysprosium', 'holmium', 'erbium', 'thulium',
    'ytterbium', 'lutetium', 'hafnium', 'tantalum', 'tungsten', 'rhenium',
    'osmium', 'iridium', 'platinum', 'gold', 'mercury', 'thallium', 'lead',
    'bismuth', 'polonium', 'astatine', 'radon', 'francium', 'radium',
    'actinium', 'thorium', 'protactinium', 'uranium', 'neptunium', 'plutonium',
    'americium', 'curium', 'berkelium', 'californium', 'einsteinium', 'fermium',
    'mendelevium', 'nobelium', 'lawrencium', 'rutherfordium', 'dubnium',
    'seaborgium', 'bohrium', 'hassium', 'meitnerium', 'darmstadtium',
    'roentgenium', 'copernicium', 'nihonium', 'flerovium', 'moscovium',
    'livermorium', 'tennessine', 'oganesson')

_SYMBOLS = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
    'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
    'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',
    'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db',
    'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og')

#: Dictionary mapping atomic number to element symbol (0 is the neutron)
ATOMIC_SYMBOL = {0: 'n', **{z: s for z, s in enumerate(_SYMBOLS, start=1)}}
#: Dictionary mapping element symbol to atomic number
ATOMIC_NUMBER = {s: z for z, s in ATOMIC_SYMBOL.items()}
#: Dictionary mapping lower-case element name to element symbol
ELEMENT_SYMBOL = dict(zip(_ELEMENTS, _SYMBOLS))
ELEMENT_SYMBOL['aluminium'] = 'Al'
ELEMENT_SYMBOL['caesium'] = 'Cs'
ELEMENT_SYMBOL['sulphur'] = 'S'

_GNDS_NAME_RE = re.compile(r'([A-Zn][a-z]*)(\d+)((?:_[em]\d+)?)')

# Used in atomic_mass function as a cache
_ATOMIC_MASS: dict[str, float] = {}


def zam(name):
    """Return tuple of (atomic number, mass number, metastable state).

    Parameters
    ----------
    name : str
        Name of nuclide using GNDS convention, e.g., 'Am242_m1'
    """
    try:
        symbol, A, state = _GNDS_NAME_RE.fullmatch(name).groups()
    except (AttributeError, TypeError):
        raise ValueError(f"'{name}' does not appear to be a nuclide name in "
                         "GNDS format")
    if symbol not in ATOMIC_NUMBER:
        raise ValueError(f"'{symbol}' is not a recognized element symbol")
    metastable = int(state[2:]) if state else 0
    return (ATOMIC_NUMBER[symbol], int(A), metastable)


def gnds_name(Z, A, m=0):
    """Return nuclide name using GNDS convention, e.g., 'Am242_m1'."""
    if m > 0:
        return f'{ATOMIC_SYMBOL[Z]}{A}_m{m}'
    return f'{ATOMIC_SYMBOL[Z]}{A}'


def _get_element_symbol(element: str) -> str:
    if len(element) > 2:
        symbol = ELEMENT_SYMBOL.get(element.lower())
        if symbol is None:
            raise ValueError(f'Element name "{element}" not recognized')
        return symbol
    return element


def isotopes(element: str) -> list[tuple[str, float]]:
    """Return naturally occurring isotopes and their abundances.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'H') or name (e.g., 'helium')

    Returns
    -------
    list
        A list of tuples of (isotope, abundance)
    """
    element = _get_element_symbol(element)
    pattern = re.compile(rf'{element}\d+')
    return [kv for kv in NATURAL_ABUNDANCE.items() if pattern.fullmatch(kv[0])]


def atomic_mass(isotope):
    """Return atomic mass of isotope in atomic mass units.

    Atomic mass data comes from the Atomic Mass Evaluation 2020.

    Parameters
    ----------
    isotope : str
        Name of isotope, e.g., 'Pu239'
    """
    if not _ATOMIC_MASS:
        mass_file = os.path.join(os.path.dirname(__file__), 'mass_1.mas20.txt')
        with open(mass_file, 'r') as ame:
            # Data starts at line 37
            for line in itertools.islice(ame, 36, None):
                name = f'{line[20:22].strip()}{int(line[16:19])}'
                mass = float(line[106:109]) + 1e-6*float(
                    line[110:116] + '.' + line[117:123])
                _ATOMIC_MASS[name.lower()] = mass

        # Natural element "isotopes" found in some libraries (e.g. C0)
        for element in ['C', 'Zn', 'Pt', 'Os', 'Tl', 'V']:
            isotope_zero = element.lower() + '0'
            _ATOMIC_MASS[isotope_zero] = 0.
            for iso, abundance in isotopes(element):
                _ATOMIC_MASS[isotope_zero] += abundance * _ATOMIC_MASS[iso.lower()]

    # Get rid of metastable information
    if '_' in isotope:
        isotope = isotope[:isotope.find('_')]
    return _ATOMIC_MASS[isotope.lower()]


def atomic_weight(element):
    """Return atomic weight of an element in atomic mass units.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'H') or name (e.g., 'helium')
    """
    weight = 0.
    for nuclide, abundance in isotopes(element):
        weight += atomic_mass(nuclide) * abundance
    if weight > 0.:
        return weight
    raise ValueError(f"No naturally-occurring isotopes for element '{element}'.")
