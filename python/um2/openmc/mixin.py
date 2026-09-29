"""Automatic ID management (compatible with :mod:`openmc.mixin`)."""

from __future__ import annotations

from numbers import Integral
from warnings import warn

import numpy as np

from . import checkvalue as cv


class EqualityMixin:
    """Provides a generic __eq__ that compares instance dictionaries."""

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if not np.array_equal(value, other.__dict__.get(key)):
                    return False
            elif value != other.__dict__.get(key):
                return False
        return True


class IDWarning(UserWarning):
    pass


class IDManagerMixin:
    """Gives subclasses a unique, automatically assigned 'id' property.

    Each class that manages its own IDs must define the class variables
    'next_id' and 'used_ids'.
    """

    min_id = 0

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, uid):
        # Find the class in the MRO that holds next_id/used_ids
        for cls in type(self).__mro__:
            if 'next_id' in cls.__dict__:
                break
        if uid is None:
            while cls.next_id in cls.used_ids:
                cls.next_id += 1
            self._id = cls.next_id
            cls.used_ids.add(cls.next_id)
        else:
            name = cls.__name__
            cv.check_type(f'{name} ID', uid, Integral)
            cv.check_greater_than(f'{name} ID', uid, cls.min_id, equality=True)
            if uid in cls.used_ids:
                warn(f'Another {name} instance already exists with id={uid}.',
                     IDWarning)
            else:
                cls.used_ids.add(uid)
            self._id = uid

    @classmethod
    def reset_ids(cls):
        """Reset counters"""
        cls.used_ids.clear()
        cls.next_id = 1


def _id_classes():
    return [cls for cls in IDManagerMixin.__subclasses__() if 'next_id' in cls.__dict__]


def reset_auto_ids():
    """Reset counters for all auto-generated IDs"""
    for cls in _id_classes():
        cls.reset_ids()


def reserve_ids(ids, cls=None):
    """Reserve a set of IDs that won't be used for auto-generated IDs."""
    classes = _id_classes() if cls is None else [cls]
    for c in classes:
        c.used_ids |= set(ids)


def set_auto_id(next_id):
    """Set the next ID for auto-generated IDs."""
    for cls in _id_classes():
        cls.next_id = next_id
