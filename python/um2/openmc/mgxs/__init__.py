"""Multigroup cross section utilities (subset of :mod:`openmc.mgxs`).

Only :class:`EnergyGroups` and the named group structures are supported. The
tally-based MGXS generation classes (e.g. :class:`openmc.mgxs.Library`) require
running OpenMC and are provided as stubs that raise
:class:`um2.errors.UM2NotSupportedError` when they would need transport results.
"""

from __future__ import annotations

import numpy as np

from ._group_structures import GROUP_STRUCTURES_DATA
from ._stubs import (MGXS, Library, AbsorptionXS, CaptureXS, Chi, FissionXS,
                     KappaFissionXS, MultiplicityMatrixXS, NuFissionMatrixXS,
                     ScatterMatrixXS, ScatterXS, TotalXS, TransportXS)
from .groups import EnergyGroups

#: Dictionary of commonly used energy group structures (from OpenMC)
GROUP_STRUCTURES = {name: np.array(edges) for name, edges in
                    GROUP_STRUCTURES_DATA.items()}

MAX_DELAYED_GROUPS = 8

# Constants used by openmc.XSdata
REPRESENTATION_ISOTROPIC = 'isotropic'
REPRESENTATION_ANGLE = 'angle'
SCATTER_TABULAR = 'tabular'
SCATTER_LEGENDRE = 'legendre'
SCATTER_HISTOGRAM = 'histogram'

__all__ = ['EnergyGroups', 'GROUP_STRUCTURES', 'Library', 'MGXS', 'TotalXS',
           'TransportXS', 'AbsorptionXS', 'CaptureXS', 'FissionXS',
           'KappaFissionXS', 'ScatterXS', 'ScatterMatrixXS', 'MultiplicityMatrixXS',
           'NuFissionMatrixXS', 'Chi']
