"""Energy group structures (compatible with :class:`openmc.mgxs.EnergyGroups`)."""

import copy
from collections.abc import Iterable
from numbers import Real

import numpy as np

from .. import checkvalue as cv


class EnergyGroups:
    """An energy groups structure used for multi-group cross-sections.

    Parameters
    ----------
    group_edges : Iterable of Real or str
        The energy group boundaries in [eV] or the name of the group structure
        (Must be a valid key in the openmc.mgxs.GROUP_STRUCTURES dictionary).
    """

    def __init__(self, group_edges):
        if isinstance(group_edges, str):
            from . import GROUP_STRUCTURES
            self._name = group_edges.upper()
            group_edges = GROUP_STRUCTURES[self._name]
        self.group_edges = group_edges

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is None:
            clone = type(self).__new__(type(self))
            clone._group_edges = copy.deepcopy(self.group_edges, memo)
            if hasattr(self, '_name'):
                clone._name = self._name
            memo[id(self)] = clone
            return clone
        return existing

    def __eq__(self, other):
        if not isinstance(other, EnergyGroups):
            return False
        if self.num_groups != other.num_groups:
            return False
        return bool(np.allclose(self.group_edges, other.group_edges))

    def __hash__(self):
        return hash(tuple(self.group_edges))

    def __repr__(self):
        if hasattr(self, '_name'):
            return f'<EnergyGroups: {self.num_groups} groups ({self._name})>'
        return f'<EnergyGroups: {self.num_groups} groups>'

    @property
    def group_edges(self):
        return self._group_edges

    @group_edges.setter
    def group_edges(self, edges):
        cv.check_type('group edges', edges, Iterable, Real)
        cv.check_increasing('group edges', edges)
        cv.check_greater_than('number of group edges', len(edges), 1)
        self._group_edges = np.array(edges, dtype=float)

    @property
    def num_groups(self):
        return len(self.group_edges) - 1

    def get_group(self, energy):
        """Returns the energy group in which the given energy resides.

        Group 1 is the highest energy group.
        """
        index = np.where(self.group_edges > energy)[0][0]
        return self.num_groups - index + 1

    def get_group_bounds(self, group):
        """Returns the energy boundaries for the energy group of interest."""
        cv.check_greater_than('group', group, 0)
        cv.check_less_than('group', group, self.num_groups, equality=True)
        lower = self.group_edges[self.num_groups-group]
        upper = self.group_edges[self.num_groups-group+1]
        return lower, upper

    def get_group_indices(self, groups='all'):
        """Returns the array indices for one or more energy groups."""
        if groups == 'all':
            return np.arange(self.num_groups)
        indices = np.zeros(len(groups), dtype=int)
        for i, group in enumerate(groups):
            cv.check_greater_than('group', group, 0)
            cv.check_less_than('group', group, self.num_groups, equality=True)
            indices[i] = group - 1
        return indices

    def get_condensed_groups(self, coarse_groups):
        """Return a coarsened version of this EnergyGroups object."""
        cv.check_type('group edges', coarse_groups, Iterable)
        for group in coarse_groups:
            cv.check_type('group edges', group, Iterable)
            cv.check_length('group edges', group, 2)
            cv.check_greater_than('lower group', group[0], 1, True)
            cv.check_less_than('lower group', group[0], self.num_groups, True)
            cv.check_greater_than('upper group', group[0], 1, True)
            cv.check_less_than('upper group', group[0], self.num_groups, True)
            cv.check_less_than('lower group', group[0], group[1], False)
        group_bounds = [group[1] for group in coarse_groups]
        group_bounds.insert(0, coarse_groups[0][0])
        group_bounds = np.asarray(group_bounds)
        group_indices = np.flipud(self.num_groups - group_bounds)
        group_indices[-1] += 1
        return EnergyGroups(np.sort(self.group_edges[group_indices]))

    def can_merge(self, other):
        """Determine if energy groups can be merged with another."""
        if not isinstance(other, EnergyGroups):
            return False
        if self == other:
            return True
        return (self.group_edges[0] == other.group_edges[-1]
                or self.group_edges[-1] == other.group_edges[0])

    def merge(self, other):
        """Merge this energy groups with another."""
        if not self.can_merge(other):
            raise ValueError('Unable to merge energy groups')
        merged_groups = copy.deepcopy(self)
        merged_edges = np.unique(np.concatenate((self.group_edges, other.group_edges)))
        merged_groups.group_edges = list(sorted(merged_edges))
        return merged_groups
