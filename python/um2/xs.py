"""Multigroup cross sections: conversion between OpenMC and MPACT conventions.

UM2 uses macroscopic multigroup cross sections. These can be given as an
OpenMC multigroup library (:class:`um2.openmc.MGXSLibrary`, stored in the
OpenMC HDF5 format) or as an MPACT user cross section library (``.xsl``). This
module converts between the two:

* :func:`mpact_xs` converts an :class:`um2.openmc.XSdata` to the MPACT
  convention: absorption, nu-fission, fission, chi and (P0...PN) scattering
  matrices indexed ``[to group, from group]``. MPACT defines the total cross
  section as absorption + P0 out-scattering, so the diagonal of the P0
  scattering matrix is adjusted such that this equals the OpenMC total (which
  is typically transport-corrected).
* :func:`write_xsl` writes an MGXSLibrary as an MPACT ``.xsl`` file.
* :func:`read_xsl` reads an MPACT ``.xsl`` file as an MGXSLibrary.
* :func:`load_library` reads either format.

In both conventions, group 1 (index 0) is the highest energy group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from .errors import UM2NotSupportedError
from .openmc.mgxs import SCATTER_HISTOGRAM, SCATTER_LEGENDRE, EnergyGroups
from .openmc.mgxs_library import REPRESENTATION_ISOTROPIC, MGXSLibrary, XSdata


@dataclass
class MPACTXS:
    """Macroscopic cross sections in the MPACT convention.

    All arrays are indexed by group, with index 0 the highest energy group.
    ``scatter[l][i, j]`` is the P_l cross section for scattering from group
    ``j`` to group ``i``.
    """
    name: str
    absorption: np.ndarray
    nu_fission: np.ndarray
    fission: np.ndarray
    chi: np.ndarray
    scatter: list

    @property
    def num_groups(self):
        return len(self.absorption)

    @property
    def order(self):
        """Scattering order (number of scattering matrices - 1)."""
        return len(self.scatter) - 1

    @property
    def total(self):
        """Total cross section: absorption + P0 out-scattering."""
        return self.absorption + self.scatter[0].sum(axis=0)

    @property
    def fissile(self):
        return bool(np.any(self.fission > 0) or np.any(self.nu_fission > 0))


def _select_temperature(xsdata: XSdata, temperature):
    temps = np.asarray(xsdata.temperatures, dtype=float)
    if temperature is None:
        if len(temps) > 1:
            warnings.warn(f'XSdata "{xsdata.name}" has data at temperatures {temps}; '
                          f'using {temps[0]} K. Set the material temperature to '
                          'select a different temperature.')
        return 0
    i = int(np.argmin(np.abs(temps - temperature)))
    if not np.isclose(temps[i], temperature):
        warnings.warn(f'XSdata "{xsdata.name}" has no data at {temperature} K; using '
                      f'the nearest temperature, {temps[i]} K.')
    return i


def mpact_xs(xsdata: XSdata, temperature=None, name=None) -> MPACTXS:
    """Convert OpenMC multigroup data to the MPACT convention.

    Parameters
    ----------
    xsdata : um2.openmc.XSdata
        Isotropic (``representation='isotropic'``) macroscopic data
    temperature : float, optional
        Temperature [K]. The nearest temperature in the data set is used. If
        not given, the first temperature is used.
    name : str, optional
        Name of the cross section set. Defaults to ``xsdata.name``.

    Returns
    -------
    MPACTXS
        The cross sections, with the diagonal of the P0 scattering matrix
        adjusted such that absorption + out-scattering equals the OpenMC total.
    """
    if xsdata.representation != REPRESENTATION_ISOTROPIC:
        raise UM2NotSupportedError(f'XSdata "{xsdata.name}": only isotropic '
                                   'representations are supported')
    it = _select_temperature(xsdata, temperature)
    G = xsdata.energy_groups.num_groups

    def get(values, label, required=False):
        v = values[it]
        if v is None:
            if required:
                raise ValueError(f'XSdata "{xsdata.name}" has no {label} data at '
                                 f'{xsdata.temperatures[it]} K')
            return np.zeros(G)
        return np.asarray(v, dtype=float)

    total = get(xsdata.total, 'total', required=True)
    absorption = get(xsdata.absorption, 'absorption', required=True)
    fission = get(xsdata.fission, 'fission')

    # nu-fission is a vector [G] or a production matrix [G_in][G_out]
    nuf = xsdata.nu_fission[it]
    chi = xsdata.chi[it]
    if nuf is None and xsdata.prompt_nu_fission[it] is not None:
        nuf = xsdata.prompt_nu_fission[it]
    if nuf is None:
        nu_fission = np.zeros(G)
    else:
        nuf = np.asarray(nuf, dtype=float)
        if nuf.ndim == 2:
            nu_fission = nuf.sum(axis=1)
            if chi is None and nuf.sum() > 0:
                chi = nuf.sum(axis=0) / nuf.sum()
        else:
            nu_fission = nuf
    chi = np.zeros(G) if chi is None else np.asarray(chi, dtype=float)

    scatter = xsdata.scatter_matrix[it]
    if scatter is None:
        raise ValueError(f'XSdata "{xsdata.name}" has no scatter matrix data at '
                         f'{xsdata.temperatures[it]} K')
    scatter = np.asarray(scatter, dtype=float)   # [G_in, G_out, order]
    if xsdata.scatter_format == SCATTER_LEGENDRE:
        matrices = [scatter[:, :, l].T.copy() for l in range(scatter.shape[2])]
    elif xsdata.scatter_format == SCATTER_HISTOGRAM:
        # The P0 moment of a histogram is the sum of its bins
        matrices = [scatter.sum(axis=2).T.copy()]
    else:
        raise UM2NotSupportedError(f'XSdata "{xsdata.name}": the '
                                   f'"{xsdata.scatter_format}" scatter format is not '
                                   'supported; use legendre or histogram')

    # Transport correction: absorption + out-scattering must equal the total
    correction = total - (absorption + matrices[0].sum(axis=0))
    matrices[0][np.diag_indices(G)] += correction
    if np.any(np.diag(matrices[0]) < 0):
        warnings.warn(f'XSdata "{xsdata.name}": the in-group scattering cross '
                      'section is negative after the transport correction')
    return MPACTXS(name or xsdata.name, absorption, nu_fission, fission, chi,
                   matrices)


def from_mpact_xs(xs: MPACTXS, energy_groups: EnergyGroups,
                  temperature=294.0) -> XSdata:
    """Convert MPACT-convention cross sections to an OpenMC XSdata.

    The total cross section is absorption + P0 out-scattering.
    """
    data = XSdata(xs.name, energy_groups, temperatures=[temperature])
    data.order = xs.order
    data.set_total(xs.total, temperature)
    data.set_absorption(xs.absorption, temperature)
    if xs.fissile:
        data.set_fission(xs.fission, temperature)
        data.set_nu_fission(xs.nu_fission, temperature)
        data.set_chi(xs.chi, temperature)
    # OpenMC indexes the scattering matrix [G_in, G_out, order]
    scatter = np.stack([m.T for m in xs.scatter], axis=2)
    data.set_scatter_matrix(scatter, temperature)
    return data


# ==============================================================================
# MPACT user cross section library (.xsl)
# ==============================================================================

def group_bounds(energy_groups: EnergyGroups) -> np.ndarray:
    """Upper energy bound of each group [eV], highest energy group first, as
    used by MPACT."""
    return np.asarray(energy_groups.group_edges)[::-1][:-1]


def write_xsl(library, filename, names=None, temperature=None, title=None,
              precision=6):
    """Write a multigroup library as an MPACT user cross section library.

    Parameters
    ----------
    library : um2.openmc.MGXSLibrary or iterable of MPACTXS
        Library to write
    filename : str or path-like
        Path of the .xsl file
    names : dict, optional
        Mapping from XSdata names to the names used in the .xsl file
    temperature : float, optional
        Temperature [K] of the data to write (the nearest available temperature
        is used)
    title : str, optional
        Title (first line) of the file
    precision : int
        Number of significant digits after the decimal point
    """
    names = names or {}
    if isinstance(library, MGXSLibrary):
        bounds = group_bounds(library.energy_groups)
        sets = [mpact_xs(x, temperature, names.get(x.name, x.name))
                for x in library.xsdatas]
        default_title = f'{library.energy_groups.num_groups}-group macroscopic cross sections'
    else:
        sets = list(library)
        if not sets:
            raise ValueError('No cross section sets to write')
        bounds = None
        default_title = f'{sets[0].num_groups}-group macroscopic cross sections'
    if bounds is None:
        raise ValueError('The group bounds are required; pass an MGXSLibrary')
    G = len(bounds)
    for xs in sets:
        if xs.num_groups != G:
            raise ValueError(f'Cross section set "{xs.name}" has {xs.num_groups} '
                             f'groups, expected {G}')
        if not xs.name or any(c.isspace() for c in xs.name):
            raise ValueError(f'Cross section set name "{xs.name}" must be non-empty '
                             'and contain no whitespace')

    fmt = f'{{:.{precision}E}}'

    def row(values):
        return '  ' + ' '.join(fmt.format(v) for v in values)

    lines = [title or default_title, f' {G} {len(sets)}',
             ' ' + ' '.join(f'{b:.6E}' for b in bounds),
             '!',
             '! Generated by UM2 (um2.xs.write_xsl)',
             '! Per group: absorption nu-fission fission chi',
             '! Scattering matrices: row = to group, column = from group',
             '!']
    for xs in sets:
        lines.append(f'XSMACRO {xs.name} {xs.order}')
        for g in range(G):
            lines.append(row((xs.absorption[g], xs.nu_fission[g], xs.fission[g],
                              xs.chi[g])))
        for matrix in xs.scatter:
            for g in range(G):
                lines.append(row(matrix[g]))
        lines.append('')
    Path(filename).write_text('\n'.join(lines) + '\n')


def _tokens(lines):
    for line in lines:
        line = line.split('!', 1)[0]
        yield from line.split()


def read_xsl(filename, temperature=294.0) -> MGXSLibrary:
    """Read an MPACT user cross section library as an OpenMC MGXSLibrary.

    The lower bound of the lowest energy group is taken to be 0 eV, and the
    total cross section is absorption + P0 out-scattering.

    Parameters
    ----------
    filename : str or path-like
        Path of the .xsl file
    temperature : float
        Temperature [K] assigned to the data

    Returns
    -------
    um2.openmc.MGXSLibrary
    """
    text = Path(filename).read_text().splitlines()
    if not text:
        raise ValueError(f'{filename} is empty')
    tokens = _tokens(text[1:])

    def number(kind=float):
        try:
            tok = next(tokens)
        except StopIteration:
            raise ValueError(f'Unexpected end of file in {filename}') from None
        try:
            return kind(tok)
        except ValueError:
            raise ValueError(f'Expected a number but found "{tok}" in {filename}') from None

    G = number(int)
    num_sets = number(int)
    bounds = np.array([number() for _ in range(G)])
    if np.any(np.diff(bounds) >= 0) or bounds[-1] <= 0:
        raise ValueError(f'Group bounds in {filename} must be positive and strictly '
                         'decreasing')
    groups = EnergyGroups(np.concatenate(([0.0], bounds[::-1])))
    library = MGXSLibrary(groups)
    for tok in tokens:
        if tok == 'XSMICRO':
            raise UM2NotSupportedError(f'XSMICRO blocks are not supported ({filename})')
        if tok != 'XSMACRO':
            raise ValueError(f'Expected XSMACRO but found "{tok}" in {filename}')
        name = next(tokens)
        order = number(int)
        principal = np.array([[number() for _ in range(4)] for _ in range(G)])
        matrices = [np.array([[number() for _ in range(G)] for _ in range(G)])
                    for _ in range(order + 1)]
        xs = MPACTXS(name, principal[:, 0], principal[:, 1], principal[:, 2],
                     principal[:, 3], matrices)
        library.add_xsdata(from_mpact_xs(xs, groups, temperature))
    if len(library.xsdatas) != num_sets:
        raise ValueError(f'Expected {num_sets} cross section sets but found '
                         f'{len(library.xsdatas)} in {filename}')
    return library


def load_library(path) -> MGXSLibrary:
    """Load a multigroup library from an OpenMC HDF5 file or an MPACT .xsl
    file (detected from the file contents)."""
    path = Path(path)
    with open(path, 'rb') as f:
        is_hdf5 = f.read(8) == b'\x89HDF\r\n\x1a\n'
    if is_hdf5:
        return MGXSLibrary.from_hdf5(path)
    return read_xsl(path)
