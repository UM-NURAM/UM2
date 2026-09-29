"""Stubs for tally-based MGXS generation, which requires running OpenMC."""

from __future__ import annotations

from ...errors import UM2NotSupportedError


class _TallyBased:
    """Accepts any construction arguments and attribute assignments, but raises
    when a method that needs OpenMC transport results is called."""

    _um2_stub = True

    def __init__(self, *args, **kwargs):
        pass

    def _unsupported(self, *args, **kwargs):
        raise UM2NotSupportedError(
            f'openmc.mgxs.{type(self).__name__} generates cross sections from '
            'OpenMC tallies, which requires running OpenMC. Generate the '
            'multigroup library with OpenMC and load it with '
            'openmc.MGXSLibrary.from_hdf5 (or Materials.cross_sections) instead.')

    def __getattr__(self, name):
        # Any method (e.g. build_library, create_mg_library) needs results
        if name.startswith('__'):
            raise AttributeError(name)
        return self._unsupported


class Library(_TallyBased):
    """Stub of :class:`openmc.mgxs.Library` (not supported by UM2)."""


class MGXS(_TallyBased):
    """Stub of :class:`openmc.mgxs.MGXS` (not supported by UM2)."""


def _make(name):
    return type(name, (MGXS,), {'__doc__': f'Stub of openmc.mgxs.{name} '
                                           '(not supported by UM2).'})


TotalXS = _make('TotalXS')
TransportXS = _make('TransportXS')
AbsorptionXS = _make('AbsorptionXS')
CaptureXS = _make('CaptureXS')
FissionXS = _make('FissionXS')
KappaFissionXS = _make('KappaFissionXS')
ScatterXS = _make('ScatterXS')
ScatterMatrixXS = _make('ScatterMatrixXS')
MultiplicityMatrixXS = _make('MultiplicityMatrixXS')
NuFissionMatrixXS = _make('NuFissionMatrixXS')
Chi = _make('Chi')
