"""Multigroup cross section libraries (compatible with :class:`openmc.XSdata`
and :class:`openmc.MGXSLibrary`).

Libraries are read and written in the OpenMC multigroup HDF5 format (requires
h5py). The conversion routines between scattering representations
(``convert_representation`` and ``convert_scatter_format``) are not provided.
"""

import copy
from numbers import Integral, Real

import numpy as np

from .data import K_BOLTZMANN
from .checkvalue import (PathLike, check_filetype_version, check_greater_than,
                         check_iterable_type, check_less_than, check_type,
                         check_value)
from .mgxs import (MAX_DELAYED_GROUPS, SCATTER_HISTOGRAM, SCATTER_LEGENDRE,
                   SCATTER_TABULAR, EnergyGroups)

ROOM_TEMPERATURE_KELVIN = 294.0

# Supported incoming particle MGXS angular treatment representations
REPRESENTATION_ISOTROPIC = 'isotropic'
REPRESENTATION_ANGLE = 'angle'
_REPRESENTATIONS = {REPRESENTATION_ISOTROPIC, REPRESENTATION_ANGLE}

# Supported scattering angular distribution representations
_SCATTER_TYPES = {SCATTER_TABULAR, SCATTER_LEGENDRE, SCATTER_HISTOGRAM}

# Filetype name of the MGXS Library
_FILETYPE_MGXS_LIBRARY = 'mgxs'

# Current version of the MGXS Library Format
_VERSION_MGXS_LIBRARY = 1


def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError('h5py is required to read or write OpenMC multigroup '
                          'cross section libraries') from e
    return h5py


class XSdata:
    """A multi-group cross section data set providing all the
    multi-group data necessary for a multi-group OpenMC calculation.

    Parameters
    ----------
    name : str
        Name of the mgxs data set.
    energy_groups : openmc.mgxs.EnergyGroups
        Energy group structure
    temperatures : Iterable of float
        Temperatures (in units of Kelvin) of the provided datasets.  Defaults
        to a single temperature at 294K.
    representation : {'isotropic', 'angle'}
        Method used in generating the MGXS (isotropic or angle-dependent flux
        weighting). Defaults to 'isotropic'
    num_delayed_groups : int
        Number of delayed groups
    """


    def __init__(self, name, energy_groups, temperatures=[ROOM_TEMPERATURE_KELVIN], representation=REPRESENTATION_ISOTROPIC, num_delayed_groups=0):
        self.name = name
        self.energy_groups = energy_groups
        self.num_delayed_groups = num_delayed_groups
        self.temperatures = temperatures
        self.representation = representation
        self._atomic_weight_ratio = None
        self._fissionable = False
        self._scatter_format = SCATTER_LEGENDRE
        self._order = None
        self._num_polar = None
        self._num_azimuthal = None
        self._total = len(temperatures) * [None]
        self._absorption = len(temperatures) * [None]
        self._scatter_matrix = len(temperatures) * [None]
        self._multiplicity_matrix = len(temperatures) * [None]
        self._fission = len(temperatures) * [None]
        self._nu_fission = len(temperatures) * [None]
        self._prompt_nu_fission = len(temperatures) * [None]
        self._delayed_nu_fission = len(temperatures) * [None]
        self._kappa_fission = len(temperatures) * [None]
        self._chi = len(temperatures) * [None]
        self._chi_prompt = len(temperatures) * [None]
        self._chi_delayed = len(temperatures) * [None]
        self._beta = len(temperatures) * [None]
        self._decay_rate = len(temperatures) * [None]
        self._inverse_velocity = len(temperatures) * [None]
        self._xs_shapes = None

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is None:
            clone = type(self).__new__(type(self))
            clone._name = self.name
            clone._energy_groups = copy.deepcopy(self.energy_groups, memo)
            clone._num_delayed_groups = self.num_delayed_groups
            clone._temperatures = copy.deepcopy(self.temperatures, memo)
            clone._representation = self.representation
            clone._atomic_weight_ratio = self._atomic_weight_ratio
            clone._fissionable = self._fissionable
            clone._scatter_format = self._scatter_format
            clone._order = self._order
            clone._num_polar = self._num_polar
            clone._num_azimuthal = self._num_azimuthal
            clone._total = copy.deepcopy(self._total, memo)
            clone._absorption = copy.deepcopy(self._absorption, memo)
            clone._scatter_matrix = copy.deepcopy(self._scatter_matrix, memo)
            clone._multiplicity_matrix = copy.deepcopy(self._multiplicity_matrix, memo)
            clone._fission = copy.deepcopy(self._fission, memo)
            clone._nu_fission = copy.deepcopy(self._nu_fission, memo)
            clone._prompt_nu_fission = copy.deepcopy(self._prompt_nu_fission, memo)
            clone._delayed_nu_fission = copy.deepcopy(self._delayed_nu_fission, memo)
            clone._kappa_fission = copy.deepcopy(self._kappa_fission, memo)
            clone._chi = copy.deepcopy(self._chi, memo)
            clone._chi_prompt = copy.deepcopy(self._chi_prompt, memo)
            clone._chi_delayed = copy.deepcopy(self._chi_delayed, memo)
            clone._beta = copy.deepcopy(self._beta, memo)
            clone._decay_rate = copy.deepcopy(self._decay_rate, memo)
            clone._inverse_velocity = copy.deepcopy(self._inverse_velocity, memo)
            clone._xs_shapes = copy.deepcopy(self._xs_shapes, memo)
            memo[id(self)] = clone
            return clone
        else:
            return existing

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type('name for XSdata', name, str)
        self._name = name

    @property
    def energy_groups(self):
        return self._energy_groups

    @energy_groups.setter
    def energy_groups(self, energy_groups):
        check_type('energy_groups', energy_groups, EnergyGroups)
        if energy_groups.group_edges is None:
            msg = 'Unable to assign an EnergyGroups object with uninitialized group edges'
            raise ValueError(msg)
        self._energy_groups = energy_groups

    @property
    def num_delayed_groups(self):
        return self._num_delayed_groups

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        check_type('num_delayed_groups', num_delayed_groups, Integral)
        check_less_than('num_delayed_groups', num_delayed_groups, MAX_DELAYED_GROUPS, equality=True)
        check_greater_than('num_delayed_groups', num_delayed_groups, 0, equality=True)
        self._num_delayed_groups = num_delayed_groups

    @property
    def representation(self):
        return self._representation

    @representation.setter
    def representation(self, representation):
        check_value('representation', representation, _REPRESENTATIONS)
        self._representation = representation

    @property
    def atomic_weight_ratio(self):
        return self._atomic_weight_ratio

    @atomic_weight_ratio.setter
    def atomic_weight_ratio(self, atomic_weight_ratio):
        check_type('atomic_weight_ratio', atomic_weight_ratio, Real)
        check_greater_than('atomic_weight_ratio', atomic_weight_ratio, 0.0)
        self._atomic_weight_ratio = atomic_weight_ratio

    @property
    def fissionable(self):
        return self._fissionable

    @property
    def temperatures(self):
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        check_iterable_type('temperatures', temperatures, Real)
        self._temperatures = np.array(temperatures)

    @property
    def scatter_format(self):
        return self._scatter_format

    @scatter_format.setter
    def scatter_format(self, scatter_format):
        check_value('scatter_format', scatter_format, _SCATTER_TYPES)
        self._scatter_format = scatter_format

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        check_type('order', order, Integral)
        check_greater_than('order', order, 0, equality=True)
        self._order = order

    @property
    def num_polar(self):
        return self._num_polar

    @num_polar.setter
    def num_polar(self, num_polar):
        check_type('num_polar', num_polar, Integral)
        check_greater_than('num_polar', num_polar, 0)
        self._num_polar = num_polar

    @property
    def num_azimuthal(self):
        return self._num_azimuthal

    @num_azimuthal.setter
    def num_azimuthal(self, num_azimuthal):
        check_type('num_azimuthal', num_azimuthal, Integral)
        check_greater_than('num_azimuthal', num_azimuthal, 0)
        self._num_azimuthal = num_azimuthal

    @property
    def total(self):
        return self._total

    @property
    def absorption(self):
        return self._absorption

    @property
    def scatter_matrix(self):
        return self._scatter_matrix

    @property
    def multiplicity_matrix(self):
        return self._multiplicity_matrix

    @property
    def fission(self):
        return self._fission

    @property
    def nu_fission(self):
        return self._nu_fission

    @property
    def prompt_nu_fission(self):
        return self._prompt_nu_fission

    @property
    def delayed_nu_fission(self):
        return self._delayed_nu_fission

    @property
    def kappa_fission(self):
        return self._kappa_fission

    @property
    def chi(self):
        return self._chi

    @property
    def chi_prompt(self):
        return self._chi_prompt

    @property
    def chi_delayed(self):
        return self._chi_delayed

    @property
    def beta(self):
        return self._beta

    @property
    def decay_rate(self):
        return self._decay_rate

    @property
    def inverse_velocity(self):
        return self._inverse_velocity

    @property
    def num_orders(self):
        if self._order is None:
            raise ValueError('Order has not been set.')
        if self._scatter_format in (None, SCATTER_LEGENDRE):
            return self._order + 1
        else:
            return self._order

    @property
    def xs_shapes(self):
        if self._xs_shapes is None:
            self._xs_shapes = {}
            self._xs_shapes['[G]'] = (self.energy_groups.num_groups,)
            self._xs_shapes["[G']"] = (self.energy_groups.num_groups,)
            self._xs_shapes["[G][G']"] = (self.energy_groups.num_groups, self.energy_groups.num_groups)
            self._xs_shapes['[DG]'] = (self.num_delayed_groups,)
            self._xs_shapes['[DG][G]'] = (self.num_delayed_groups, self.energy_groups.num_groups)
            self._xs_shapes["[DG][G']"] = (self.num_delayed_groups, self.energy_groups.num_groups)
            self._xs_shapes["[DG][G][G']"] = (self.num_delayed_groups, self.energy_groups.num_groups, self.energy_groups.num_groups)
            self._xs_shapes["[G][G'][Order]"] = (self.energy_groups.num_groups, self.energy_groups.num_groups, self.num_orders)
            if self.representation == REPRESENTATION_ANGLE:
                for key, shapes in self._xs_shapes.items():
                    self._xs_shapes[key] = (self.num_polar, self.num_azimuthal) + shapes
        return self._xs_shapes

    def add_temperature(self, temperature):
        check_type('temperature', temperature, Real)
        temp_store = self.temperatures.tolist()
        temp_store.append(temperature)
        self.temperatures = temp_store
        self._total.append(None)
        self._absorption.append(None)
        self._scatter_matrix.append(None)
        self._multiplicity_matrix.append(None)
        self._fission.append(None)
        self._nu_fission.append(None)
        self._prompt_nu_fission.append(None)
        self._delayed_nu_fission.append(None)
        self._kappa_fission.append(None)
        self._chi.append(None)
        self._chi_prompt.append(None)
        self._chi_delayed.append(None)
        self._beta.append(None)
        self._decay_rate.append(None)
        self._inverse_velocity.append(None)

    def _check_temperature(self, temperature):
        check_type('temperature', temperature, Real)
        check_value('temperature', temperature, self.temperatures)

    def _temperature_index(self, temperature):
        return np.where(self.temperatures == temperature)[0][0]

    def _set_fissionable(self, array):
        if np.sum(array) > 0:
            self._fissionable = True

    def set_total(self, total, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[G]']]
        total = np.asarray(total)
        check_value('total shape', total.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._total[i] = total

    def set_absorption(self, absorption, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[G]']]
        absorption = np.asarray(absorption)
        check_value('absorption shape', absorption.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._absorption[i] = absorption

    def set_fission(self, fission, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[G]']]
        fission = np.asarray(fission)
        check_value('fission shape', fission.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._fission[i] = fission
        self._set_fissionable(fission)

    def set_kappa_fission(self, kappa_fission, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[G]']]
        kappa_fission = np.asarray(kappa_fission)
        check_value('kappa fission shape', kappa_fission.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._kappa_fission[i] = kappa_fission
        self._set_fissionable(kappa_fission)

    def set_chi(self, chi, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes["[G']"]]
        chi = np.asarray(chi)
        check_value('chi shape', chi.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._chi[i] = chi

    def set_chi_prompt(self, chi_prompt, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes["[G']"]]
        chi_prompt = np.asarray(chi_prompt)
        check_value('chi prompt shape', chi_prompt.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._chi_prompt[i] = chi_prompt

    def set_chi_delayed(self, chi_delayed, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes["[G']"], self.xs_shapes["[DG][G']"]]
        chi_delayed = np.asarray(chi_delayed)
        check_value('chi delayed shape', chi_delayed.shape, shapes)
        self._check_temperature(temperature)
        check_type('temperature', temperature, Real)
        check_value('temperature', temperature, self.temperatures)
        i = self._temperature_index(temperature)
        self._chi_delayed[i] = chi_delayed

    def set_beta(self, beta, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[DG]'], self.xs_shapes['[DG][G]']]
        beta = np.asarray(beta)
        check_value('beta shape', beta.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._beta[i] = beta

    def set_decay_rate(self, decay_rate, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[DG]']]
        decay_rate = np.asarray(decay_rate)
        check_value('decay rate shape', decay_rate.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._decay_rate[i] = decay_rate

    def set_scatter_matrix(self, scatter, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes["[G][G'][Order]"]]
        scatter = np.asarray(scatter)
        check_iterable_type('scatter', scatter, Real, max_depth=len(scatter.shape))
        check_value('scatter shape', scatter.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._scatter_matrix[i] = scatter

    def set_multiplicity_matrix(self, multiplicity, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes["[G][G']"]]
        multiplicity = np.asarray(multiplicity)
        check_iterable_type('multiplicity', multiplicity, Real, max_depth=len(multiplicity.shape))
        check_value('multiplicity shape', multiplicity.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._multiplicity_matrix[i] = multiplicity

    def set_nu_fission(self, nu_fission, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[G]'], self.xs_shapes["[G][G']"]]
        nu_fission = np.asarray(nu_fission)
        check_value('nu_fission shape', nu_fission.shape, shapes)
        check_iterable_type('nu_fission', nu_fission, Real, max_depth=len(nu_fission.shape))
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._nu_fission[i] = nu_fission
        self._set_fissionable(nu_fission)

    def set_prompt_nu_fission(self, prompt_nu_fission, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[G]'], self.xs_shapes["[G][G']"]]
        prompt_nu_fission = np.asarray(prompt_nu_fission)
        check_value('prompt_nu_fission shape', prompt_nu_fission.shape, shapes)
        check_iterable_type('prompt_nu_fission', prompt_nu_fission, Real, max_depth=len(prompt_nu_fission.shape))
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._prompt_nu_fission[i] = prompt_nu_fission
        self._set_fissionable(prompt_nu_fission)

    def set_delayed_nu_fission(self, delayed_nu_fission, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[DG][G]'], self.xs_shapes["[DG][G][G']"]]
        delayed_nu_fission = np.asarray(delayed_nu_fission)
        check_value('delayed_nu_fission shape', delayed_nu_fission.shape, shapes)
        check_iterable_type('delayed_nu_fission', delayed_nu_fission, Real, max_depth=len(delayed_nu_fission.shape))
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._delayed_nu_fission[i] = delayed_nu_fission
        self._set_fissionable(delayed_nu_fission)

    def set_inverse_velocity(self, inv_vel, temperature=ROOM_TEMPERATURE_KELVIN):
        shapes = [self.xs_shapes['[G]']]
        inv_vel = np.asarray(inv_vel)
        check_value('inverse_velocity shape', inv_vel.shape, shapes)
        self._check_temperature(temperature)
        i = self._temperature_index(temperature)
        self._inverse_velocity[i] = inv_vel

    def to_hdf5(self, file):
        grp = file.create_group(self.name)
        if self.atomic_weight_ratio is not None:
            grp.attrs['atomic_weight_ratio'] = self.atomic_weight_ratio
        if self.fissionable is not None:
            grp.attrs['fissionable'] = self.fissionable
        if self.representation is not None:
            grp.attrs['representation'] = np.bytes_(self.representation)
            if self.representation == REPRESENTATION_ANGLE:
                if self.num_azimuthal is not None:
                    grp.attrs['num_azimuthal'] = self.num_azimuthal
                if self.num_polar is not None:
                    grp.attrs['num_polar'] = self.num_polar
        grp.attrs['scatter_shape'] = np.bytes_("[G][G'][Order]")
        if self.scatter_format is not None:
            grp.attrs['scatter_format'] = np.bytes_(self.scatter_format)
        if self.order is not None:
            grp.attrs['order'] = self.order
        ktg = grp.create_group('kTs')
        for temperature in self.temperatures:
            temp_label = str(int(np.round(temperature))) + 'K'
            kT = temperature * K_BOLTZMANN
            ktg.create_dataset(temp_label, data=kT)
        for i, temperature in enumerate(self.temperatures):
            xs_grp = grp.create_group(str(int(np.round(temperature))) + 'K')
            if self._total[i] is None:
                raise ValueError('total data must be provided when writing the HDF5 library')
            xs_grp.create_dataset('total', data=self._total[i])
            if self._absorption[i] is None:
                raise ValueError('absorption data must be provided when writing the HDF5 library')
            xs_grp.create_dataset('absorption', data=self._absorption[i])
            if self.fissionable:
                if self._fission[i] is not None:
                    xs_grp.create_dataset('fission', data=self._fission[i])
                if self._kappa_fission[i] is not None:
                    xs_grp.create_dataset('kappa-fission', data=self._kappa_fission[i])
                if self._chi[i] is not None:
                    xs_grp.create_dataset('chi', data=self._chi[i])
                if self._chi_prompt[i] is not None:
                    xs_grp.create_dataset('chi-prompt', data=self._chi_prompt[i])
                if self._chi_delayed[i] is not None:
                    xs_grp.create_dataset('chi-delayed', data=self._chi_delayed[i])
                if self._nu_fission[i] is None and (self._delayed_nu_fission[i] is None or self._prompt_nu_fission[i] is None):
                    raise ValueError('nu-fission or prompt-nu-fission and delayed-nu-fission data must be provided when writing the HDF5 library')
                if self._nu_fission[i] is not None:
                    xs_grp.create_dataset('nu-fission', data=self._nu_fission[i])
                if self._prompt_nu_fission[i] is not None:
                    xs_grp.create_dataset('prompt-nu-fission', data=self._prompt_nu_fission[i])
                if self._delayed_nu_fission[i] is not None:
                    xs_grp.create_dataset('delayed-nu-fission', data=self._delayed_nu_fission[i])
                if self._beta[i] is not None:
                    xs_grp.create_dataset('beta', data=self._beta[i])
                if self._decay_rate[i] is not None:
                    xs_grp.create_dataset('decay-rate', data=self._decay_rate[i])
            if self._scatter_matrix[i] is None:
                raise ValueError('Scatter matrix must be provided when writing the HDF5 library')
            G = self.energy_groups.num_groups
            if self.representation == REPRESENTATION_ISOTROPIC:
                Np = 1
                Na = 1
            elif self.representation == REPRESENTATION_ANGLE:
                Np = self.num_polar
                Na = self.num_azimuthal
            g_out_bounds = np.zeros((Np, Na, G, 2), dtype=int)
            for p in range(Np):
                for a in range(Na):
                    for g_in in range(G):
                        if self.scatter_format == SCATTER_LEGENDRE:
                            if self.representation == REPRESENTATION_ISOTROPIC:
                                matrix = self._scatter_matrix[i][g_in, :, 0]
                            elif self.representation == REPRESENTATION_ANGLE:
                                matrix = self._scatter_matrix[i][p, a, g_in, :, 0]
                        elif self.representation == REPRESENTATION_ISOTROPIC:
                            matrix = np.sum(self._scatter_matrix[i][g_in, :, :], axis=1)
                        elif self.representation == REPRESENTATION_ANGLE:
                            matrix = np.sum(self._scatter_matrix[i][p, a, g_in, :, :], axis=1)
                        nz = np.nonzero(matrix)
                        if len(nz[0]) == 0:
                            g_out_bounds[p, a, g_in, :] = 0
                        else:
                            g_out_bounds[p, a, g_in, 0] = nz[0][0]
                            g_out_bounds[p, a, g_in, 1] = nz[0][-1]
            flat_scatt = []
            for p in range(Np):
                for a in range(Na):
                    if self.representation == REPRESENTATION_ISOTROPIC:
                        matrix = self._scatter_matrix[i][:, :, :]
                    elif self.representation == REPRESENTATION_ANGLE:
                        matrix = self._scatter_matrix[i][p, a, :, :, :]
                    for g_in in range(G):
                        for g_out in range(g_out_bounds[p, a, g_in, 0], g_out_bounds[p, a, g_in, 1] + 1):
                            for l in range(len(matrix[g_in, g_out, :])):
                                flat_scatt.append(matrix[g_in, g_out, l])
            scatt_grp = xs_grp.create_group('scatter_data')
            scatt_grp.create_dataset('scatter_matrix', data=np.array(flat_scatt))
            if self._multiplicity_matrix[i] is not None:
                flat_mult = []
                for p in range(Np):
                    for a in range(Na):
                        if self.representation == REPRESENTATION_ISOTROPIC:
                            matrix = self._multiplicity_matrix[i][:, :]
                        elif self.representation == REPRESENTATION_ANGLE:
                            matrix = self._multiplicity_matrix[i][p, a, :, :]
                        for g_in in range(G):
                            for g_out in range(g_out_bounds[p, a, g_in, 0], g_out_bounds[p, a, g_in, 1] + 1):
                                flat_mult.append(matrix[g_in, g_out])
                scatt_grp.create_dataset('multiplicity_matrix', data=np.array(flat_mult))
            g_out_bounds[:, :, :, :] += 1
            if self.representation == REPRESENTATION_ISOTROPIC:
                scatt_grp.create_dataset('g_min', data=g_out_bounds[0, 0, :, 0])
                scatt_grp.create_dataset('g_max', data=g_out_bounds[0, 0, :, 1])
            elif self.representation == REPRESENTATION_ANGLE:
                scatt_grp.create_dataset('g_min', data=g_out_bounds[:, :, :, 0])
                scatt_grp.create_dataset('g_max', data=g_out_bounds[:, :, :, 1])
            if self._inverse_velocity[i] is not None:
                xs_grp.create_dataset('inverse-velocity', data=self._inverse_velocity[i])

    @classmethod
    def from_hdf5(cls, group, name, energy_groups, num_delayed_groups):
        subgroups = group.keys()
        temperatures = []
        for subgroup in subgroups:
            if subgroup != 'kTs':
                temperatures.append(subgroup)
        kTs_group = group['kTs']
        float_temperatures = []
        for temperature in temperatures:
            kT = kTs_group[temperature][()]
            float_temperatures.append(kT / K_BOLTZMANN)
        attrs = group.attrs.keys()
        if 'representation' in attrs:
            representation = group.attrs['representation'].decode()
        else:
            representation = REPRESENTATION_ISOTROPIC
        data = cls(name, energy_groups, float_temperatures, representation, num_delayed_groups)
        if 'scatter_format' in attrs:
            data.scatter_format = group.attrs['scatter_format'].decode()
        if 'atomic_weight_ratio' in attrs:
            data.atomic_weight_ratio = group.attrs['atomic_weight_ratio']
        if 'order' in attrs:
            data.order = group.attrs['order']
        if data.representation == REPRESENTATION_ANGLE:
            data.num_azimuthal = group.attrs['num_azimuthal']
            data.num_polar = group.attrs['num_polar']
        for temp, float_temp in zip(temperatures, float_temperatures):
            xs_types = ['total', 'absorption', 'fission', 'kappa-fission', 'chi', 'chi-prompt', 'chi-delayed', 'nu-fission', 'prompt-nu-fission', 'delayed-nu-fission', 'beta', 'decay-rate', 'inverse-velocity']
            temperature_group = group[temp]
            for xs_type in xs_types:
                set_func = 'set_' + xs_type.replace(' ', '_').replace('-', '_')
                if xs_type in temperature_group:
                    getattr(data, set_func)(temperature_group[xs_type][()], float_temp)
            scatt_group = temperature_group['scatter_data']
            g_max = scatt_group['g_max']
            g_min = scatt_group['g_min']
            flat_scatter = scatt_group['scatter_matrix'][()]
            scatter_matrix = np.zeros(data.xs_shapes["[G][G'][Order]"])
            G = data.energy_groups.num_groups
            if data.representation == REPRESENTATION_ISOTROPIC:
                Np = 1
                Na = 1
            elif data.representation == REPRESENTATION_ANGLE:
                Np = data.num_polar
                Na = data.num_azimuthal
            flat_index = 0
            for p in range(Np):
                for a in range(Na):
                    for g_in in range(G):
                        if data.representation == REPRESENTATION_ISOTROPIC:
                            g_mins = g_min[g_in]
                            g_maxs = g_max[g_in]
                        elif data.representation == REPRESENTATION_ANGLE:
                            g_mins = g_min[p, a, g_in]
                            g_maxs = g_max[p, a, g_in]
                        for g_out in range(g_mins - 1, g_maxs):
                            for ang in range(data.num_orders):
                                if data.representation == REPRESENTATION_ISOTROPIC:
                                    scatter_matrix[g_in, g_out, ang] = flat_scatter[flat_index]
                                elif data.representation == REPRESENTATION_ANGLE:
                                    scatter_matrix[p, a, g_in, g_out, ang] = flat_scatter[flat_index]
                                flat_index += 1
            data.set_scatter_matrix(scatter_matrix, float_temp)
            if 'multiplicity_matrix' in scatt_group:
                flat_mult = scatt_group['multiplicity_matrix'][()]
                mult_matrix = np.zeros(data.xs_shapes["[G][G']"])
                flat_index = 0
                for p in range(Np):
                    for a in range(Na):
                        for g_in in range(G):
                            if data.representation == REPRESENTATION_ISOTROPIC:
                                g_mins = g_min[g_in]
                                g_maxs = g_max[g_in]
                            elif data.representation == REPRESENTATION_ANGLE:
                                g_mins = g_min[p, a, g_in]
                                g_maxs = g_max[p, a, g_in]
                            for g_out in range(g_mins - 1, g_maxs):
                                if data.representation == REPRESENTATION_ISOTROPIC:
                                    mult_matrix[g_in, g_out] = flat_mult[flat_index]
                                elif data.representation == REPRESENTATION_ANGLE:
                                    mult_matrix[p, a, g_in, g_out] = flat_mult[flat_index]
                                flat_index += 1
                data.set_multiplicity_matrix(mult_matrix, float_temp)
        return data

class MGXSLibrary:
    """Multi-Group Cross Sections file used for an OpenMC simulation.

    Parameters
    ----------
    energy_groups : openmc.mgxs.EnergyGroups
        Energy group structure.
    num_delayed_groups : int
        Num delayed groups
    """

    def __init__(self, energy_groups, num_delayed_groups=0):
        self.energy_groups = energy_groups
        self.num_delayed_groups = num_delayed_groups
        self._xsdatas = []

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is None:
            clone = type(self).__new__(type(self))
            clone._energy_groups = copy.deepcopy(self.energy_groups, memo)
            clone._num_delayed_groups = self.num_delayed_groups
            clone._xsdatas = copy.deepcopy(self.xsdatas, memo)
            memo[id(self)] = clone
            return clone
        else:
            return existing

    @property
    def energy_groups(self):
        return self._energy_groups

    @energy_groups.setter
    def energy_groups(self, energy_groups):
        check_type('energy groups', energy_groups, EnergyGroups)
        self._energy_groups = energy_groups

    @property
    def num_delayed_groups(self):
        return self._num_delayed_groups

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        check_type('num_delayed_groups', num_delayed_groups, Integral)
        check_greater_than('num_delayed_groups', num_delayed_groups, 0, equality=True)
        check_less_than('num_delayed_groups', num_delayed_groups, MAX_DELAYED_GROUPS, equality=True)
        self._num_delayed_groups = num_delayed_groups

    @property
    def xsdatas(self):
        return self._xsdatas

    @property
    def names(self):
        return [xsdata.name for xsdata in self.xsdatas]

    def add_xsdata(self, xsdata):
        if not isinstance(xsdata, XSdata):
            msg = f'Unable to add a non-XSdata "{xsdata}" to the MGXSLibrary instance'
            raise ValueError(msg)
        if xsdata.energy_groups != self._energy_groups:
            msg = 'Energy groups of XSdata do not match that of MGXSLibrary.'
            raise ValueError(msg)
        self._xsdatas.append(xsdata)

    def add_xsdatas(self, xsdatas):
        check_iterable_type('xsdatas', xsdatas, XSdata)
        for xsdata in xsdatas:
            self.add_xsdata(xsdata)

    def remove_xsdata(self, xsdata):
        if not isinstance(xsdata, XSdata):
            msg = f'Unable to remove a non-XSdata "{xsdata}" from the MGXSLibrary instance'
            raise ValueError(msg)
        self._xsdatas.remove(xsdata)

    def get_by_name(self, name):
        check_type('name', name, str)
        result = None
        for xsdata in self.xsdatas:
            if name == xsdata.name:
                result = xsdata
        return result

    def export_to_hdf5(self, filename='mgxs.h5', libver='earliest'):
        check_type('filename', filename, PathLike)
        file = _h5py().File(filename, 'w', libver=libver)
        file.attrs['filetype'] = np.bytes_(_FILETYPE_MGXS_LIBRARY)
        file.attrs['version'] = [_VERSION_MGXS_LIBRARY, 0]
        file.attrs['energy_groups'] = self.energy_groups.num_groups
        file.attrs['delayed_groups'] = self.num_delayed_groups
        file.attrs['group structure'] = self.energy_groups.group_edges
        for xsdata in self._xsdatas:
            xsdata.to_hdf5(file)
        file.close()

    @classmethod
    def from_hdf5(cls, filename=None):
        if filename is None:
            from .config import config
            filename = config.get_path('mg_cross_sections')
        if filename is None:
            raise ValueError("Either path or openmc.config['mg_cross_sections']must be set")
        check_type('filename', filename, PathLike)
        with _h5py().File(filename, 'r') as file:
            check_filetype_version(file, _FILETYPE_MGXS_LIBRARY, _VERSION_MGXS_LIBRARY)
            group_structure = file.attrs['group structure']
            num_delayed_groups = file.attrs['delayed_groups']
            energy_groups = EnergyGroups(group_structure)
            data = cls(energy_groups, num_delayed_groups)
            for group_name, group in file.items():
                data.add_xsdata(XSdata.from_hdf5(group, group_name, energy_groups, num_delayed_groups))
        return data
