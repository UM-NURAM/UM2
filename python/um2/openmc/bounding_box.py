"""Axis-aligned bounding boxes (compatible with :class:`openmc.BoundingBox`)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .checkvalue import check_length


class BoundingBox:
    """Axis-aligned bounding box.

    Parameters
    ----------
    lower_left : iterable of float
        The x, y, z coordinates of the lower left corner of the bounding box
    upper_right : iterable of float
        The x, y, z coordinates of the upper right corner of the bounding box
    """

    def __init__(self, lower_left: Iterable[float], upper_right: Iterable[float]):
        check_length("lower_left", lower_left, 3, 3)
        check_length("upper_right", upper_right, 3, 3)
        self._bounds = np.asarray([lower_left, upper_right], dtype=float)

    def __repr__(self) -> str:
        return "BoundingBox(lower_left={}, upper_right={})".format(
            tuple(float(x) for x in self.lower_left),
            tuple(float(x) for x in self.upper_right))

    def __getitem__(self, key) -> np.ndarray:
        return self._bounds[key]

    def __len__(self):
        return 2

    def __setitem__(self, key, val):
        self._bounds[key] = val

    def __iand__(self, other: BoundingBox) -> BoundingBox:
        self.lower_left = np.maximum(self.lower_left, other.lower_left)
        self.upper_right = np.minimum(self.upper_right, other.upper_right)
        return self

    def __and__(self, other: BoundingBox) -> BoundingBox:
        new = BoundingBox(*self)
        new &= other
        return new

    def __ior__(self, other: BoundingBox) -> BoundingBox:
        self.lower_left = np.minimum(self.lower_left, other.lower_left)
        self.upper_right = np.maximum(self.upper_right, other.upper_right)
        return self

    def __or__(self, other: BoundingBox) -> BoundingBox:
        new = BoundingBox(*self)
        new |= other
        return new

    def __contains__(self, other):
        if isinstance(other, (tuple, list, np.ndarray)):
            check_length("Point", other, 3, 3)
            return all(other > self.lower_left) and all(other < self.upper_right)
        if isinstance(other, BoundingBox):
            return all(p in self for p in [other.lower_left, other.upper_right])
        raise TypeError(f"Unable to determine if {other} is in the bounding box. "
                        f"Expected a tuple or a bounding box, but {type(other)} given")

    @property
    def center(self) -> np.ndarray:
        return (self[0] + self[1]) / 2

    @property
    def lower_left(self) -> np.ndarray:
        return self[0]

    @lower_left.setter
    def lower_left(self, llc):
        check_length('lower_left', llc, 3, 3)
        self[0] = llc

    @property
    def upper_right(self) -> np.ndarray:
        return self[1]

    @upper_right.setter
    def upper_right(self, urc):
        check_length('upper_right', urc, 3, 3)
        self[1] = urc

    @property
    def volume(self) -> float:
        return np.abs(np.prod(self[1] - self[0]))

    @property
    def extent(self):
        ll, ur = self.lower_left, self.upper_right
        return {'xy': (ll[0], ur[0], ll[1], ur[1]),
                'xz': (ll[0], ur[0], ll[2], ur[2]),
                'yz': (ll[1], ur[1], ll[2], ur[2])}

    @property
    def width(self):
        return self.upper_right - self.lower_left

    def expand(self, padding_distance: float, inplace: bool = False) -> BoundingBox:
        """Expand the bounding box by a padding distance in all directions."""
        if inplace:
            self[0] -= padding_distance
            self[1] += padding_distance
            return self
        return BoundingBox(self[0] - padding_distance, self[1] + padding_distance)

    @classmethod
    def infinite(cls) -> BoundingBox:
        """Create an infinite box."""
        infs = np.full((3,), np.inf)
        return cls(-infs, infs)
