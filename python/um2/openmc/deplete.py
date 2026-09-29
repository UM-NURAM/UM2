"""Stub of :mod:`openmc.deplete` (depletion requires running OpenMC)."""

from __future__ import annotations

from ..errors import UM2NotSupportedError


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    raise UM2NotSupportedError(f'openmc.deplete.{name}: depletion requires running '
                               'OpenMC and is not supported by UM2.')
