"""Subset of :mod:`openmc.data` needed to define materials."""

from __future__ import annotations

from .data import (ATOMIC_NUMBER, ATOMIC_SYMBOL, AVOGADRO, ELEMENT_SYMBOL,
                   EV_PER_MEV, JOULE_PER_EV, K_BOLTZMANN, NATURAL_ABUNDANCE,
                   NEUTRON_MASS, atomic_mass, atomic_weight, gnds_name,
                   isotopes, zam)

__all__ = ['ATOMIC_NUMBER', 'ATOMIC_SYMBOL', 'AVOGADRO', 'ELEMENT_SYMBOL',
           'EV_PER_MEV', 'JOULE_PER_EV', 'K_BOLTZMANN', 'NATURAL_ABUNDANCE',
           'NEUTRON_MASS', 'atomic_mass', 'atomic_weight', 'gnds_name',
           'isotopes', 'zam']
