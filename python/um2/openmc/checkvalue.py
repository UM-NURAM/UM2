"""Input validation helpers (compatible with :mod:`openmc.checkvalue`)."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable

import numpy as np

PathLike = str | os.PathLike


def _type_names(types):
    if isinstance(types, Iterable):
        return ', '.join(t.__name__ for t in types)
    return types.__name__


def check_type(name, value, expected_type, expected_iter_type=None, *, none_ok=False):
    """Ensure that an object is of an expected type (and optionally that each
    item of an iterable is of an expected type)."""
    if none_ok and value is None:
        return
    if not isinstance(value, expected_type):
        raise TypeError(f'Unable to set "{name}" to "{value}" which is not of '
                        f'type "{_type_names(expected_type)}"')
    if expected_iter_type:
        if isinstance(value, np.ndarray):
            if not issubclass(value.dtype.type, expected_iter_type):
                raise TypeError(f'Unable to set "{name}" to "{value}" since each '
                                f'item must be of type '
                                f'"{_type_names(expected_iter_type)}"')
            return
        for item in value:
            if not isinstance(item, expected_iter_type):
                raise TypeError(f'Unable to set "{name}" to "{value}" since each '
                                f'item must be of type '
                                f'"{_type_names(expected_iter_type)}"')


def check_iterable_type(name, value, expected_type, min_depth=1, max_depth=1):
    """Ensure that an object is an iterable (of iterables, ...) whose leaves are
    of an expected type and that are nested between min_depth and max_depth
    levels deep."""
    def visit(item, depth, index):
        if isinstance(item, expected_type):
            if depth < min_depth:
                raise TypeError(f'Error setting "{name}": The item at {index} does '
                                f'not meet the minimum depth of {min_depth}')
            return
        if isinstance(item, Iterable) and not isinstance(item, str):
            if depth + 1 > max_depth:
                raise TypeError(f'Error setting {name}: Found an iterable at '
                                f'{index}, items in that iterable exceed the '
                                f'maximum depth of {max_depth}')
            for i, sub in enumerate(item):
                visit(sub, depth + 1, index + [i])
            return
        raise TypeError(f'Error setting {name}: Items must be of type '
                        f'"{expected_type.__name__}", but item at {index} is of '
                        f'type "{type(item).__name__}"')

    for i, item in enumerate(value):
        visit(item, 1, [i])


def check_length(name, value, length_min, length_max=None):
    """Ensure that a sized object has a length within a given range."""
    if length_max is None:
        if len(value) < length_min:
            raise ValueError(f'Unable to set "{name}" to "{value}" since it must '
                             f'be at least of length "{length_min}"')
    elif not length_min <= len(value) <= length_max:
        if length_min == length_max:
            raise ValueError(f'Unable to set "{name}" to "{value}" since it must '
                             f'be of length "{length_min}"')
        raise ValueError(f'Unable to set "{name}" to "{value}" since it must '
                         f'have length between "{length_min}" and "{length_max}"')


def check_increasing(name: str, value, equality: bool = False):
    """Ensure that a sequence is (strictly, unless equality) increasing."""
    diff = np.diff(value)
    if equality and not np.all(diff >= 0.0):
        raise ValueError(f'Unable to set "{name}" to "{value}" since its '
                         'elements must be increasing.')
    if not equality and not np.all(diff > 0.0):
        raise ValueError(f'Unable to set "{name}" to "{value}" since its '
                         'elements must be strictly increasing.')


def check_value(name, value, accepted_values):
    """Ensure that an object's value is one of a set of accepted values."""
    if value not in accepted_values:
        raise ValueError(f'Unable to set "{name}" to "{value}" since it is not '
                         f'in "{accepted_values}"')


def check_less_than(name, value, maximum, equality=False):
    """Ensure that a value is less than (or equal to) a maximum."""
    if value > maximum or (not equality and value == maximum):
        qualifier = '' if equality else ' or equal to'
        raise ValueError(f'Unable to set "{name}" to "{value}" since it is '
                         f'greater than{qualifier} "{maximum}"')


def check_greater_than(name, value, minimum, equality=False):
    """Ensure that a value is greater than (or equal to) a minimum."""
    if value < minimum or (not equality and value == minimum):
        qualifier = '' if equality else ' or equal to'
        raise ValueError(f'Unable to set "{name}" to "{value}" since it is '
                         f'less than{qualifier} "{minimum}"')


def check_filetype_version(obj, expected_type, expected_version):
    """Check the filetype and major version of an HDF5 file object."""
    try:
        this_filetype = obj.attrs['filetype']
        if isinstance(this_filetype, bytes):
            this_filetype = this_filetype.decode()
        this_version = obj.attrs['version']
    except (AttributeError, KeyError):
        raise IOError(f'Could not read {obj.filename} file. This most likely '
                      'means the file was produced by a different version of '
                      'OpenMC than the one you are using.')
    if this_filetype != expected_type:
        raise IOError(f'{obj.filename} is not a {expected_type} file.')
    if this_version[0] != expected_version:
        raise IOError(f'{this_filetype} file has a version of '
                      f'{".".join(str(v) for v in this_version)} which is not '
                      f'consistent with the expected version, {expected_version}')


class CheckedList(list):
    """A list that checks the type of each item that is added to it."""

    def __init__(self, expected_type, name, items=None):
        super().__init__()
        self.expected_type = expected_type
        self.name = name
        if items is not None:
            for item in items:
                self.append(item)

    def __add__(self, other):
        new_instance = copy.copy(self)
        new_instance += other
        return new_instance

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        check_type('CheckedList add operand', other, Iterable, self.expected_type)
        for item in other:
            self.append(item)
        return self

    def append(self, item):
        check_type(self.name, item, self.expected_type)
        super().append(item)

    def insert(self, index, item):
        check_type(self.name, item, self.expected_type)
        super().insert(index, item)
