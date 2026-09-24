"""Regions of space (compatible with :mod:`openmc.region`)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from copy import deepcopy

import numpy as np

from .bounding_box import BoundingBox


class Region(ABC):
    """Region of space that can be assigned to a cell.

    Region is an abstract base class that is inherited by
    :class:`openmc.Halfspace`, :class:`openmc.Intersection`,
    :class:`openmc.Union`, and :class:`openmc.Complement`. Each of those
    respective classes are typically not instantiated directly but rather are
    created through operators of the Surface and Region classes.
    """

    def __and__(self, other):
        return Intersection((self, other))

    def __or__(self, other):
        return Union((self, other))

    @abstractmethod
    def __invert__(self) -> Region:
        pass

    @abstractmethod
    def __contains__(self, point):
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> BoundingBox:
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return str(self) == str(other)

    def __hash__(self):
        return id(self)

    def get_surfaces(self, surfaces=None):
        """Recursively find all surfaces referenced by a region and return them

        Returns
        -------
        surfaces : dict
            Dictionary mapping surface IDs to :class:`openmc.Surface` instances
        """
        if surfaces is None:
            surfaces = {}
        for region in self:
            surfaces = region.get_surfaces(surfaces)
        return surfaces

    def remove_redundant_surfaces(self, redundant_surfaces):
        """Recursively remove all redundant surfaces referenced by this region"""
        for region in self:
            region.remove_redundant_surfaces(redundant_surfaces)

    @staticmethod
    def from_expression(expression, surfaces):
        """Generate a region given an infix expression.

        Parameters
        ----------
        expression : str
            Boolean expression relating surface half-spaces. The possible
            operators are union '|', intersection ' ', and complement '~'. For
            example, '(1 -2) | 3 ~(4 -5)'.
        surfaces : dict
            Dictionary whose keys are surface IDs that appear in the Boolean
            expression and whose values are Surface objects.
        """
        expression = expression.strip()
        i = 0
        i_start = -1
        tokens = []
        while i < len(expression):
            if expression[i] in '()|~ ':
                if i_start >= 0:
                    j = int(expression[i_start:i])
                    tokens.append(-surfaces[abs(j)] if j < 0 else +surfaces[abs(j)])
                    if expression[i] == '(':
                        tokens.append(' ')
                if expression[i] in '()|~':
                    tokens.append(expression[i])
                    if expression[i:i+2] == ')(':
                        tokens.append(' ')
                else:
                    while expression[i+1] == ' ':
                        i += 1
                    if (i_start >= 0 or tokens[-1] == ')') and \
                            expression[i+1] not in ')|':
                        tokens.append(' ')
                i_start = -1
            else:
                if expression[i] not in '-+0123456789':
                    raise SyntaxError(f"Invalid character '{expression[i]}' in "
                                      "expression")
                if i_start < 0:
                    i_start = i
            i += 1
        if i_start >= 0:
            j = int(expression[i_start:])
            tokens.append(-surfaces[abs(j)] if j < 0 else +surfaces[abs(j)])

        def can_be_combined(region):
            return isinstance(region, Complement) or hasattr(region, 'surface')

        def apply_operator(output, operator):
            r2 = output.pop()
            if operator == ' ':
                r1 = output.pop()
                if isinstance(r1, Intersection):
                    r1 &= r2
                    output.append(r1)
                elif isinstance(r2, Intersection) and can_be_combined(r1):
                    r2.insert(0, r1)
                    output.append(r2)
                else:
                    output.append(r1 & r2)
            elif operator == '|':
                r1 = output.pop()
                if isinstance(r1, Union):
                    r1 |= r2
                    output.append(r1)
                elif isinstance(r2, Union) and can_be_combined(r1):
                    r2.insert(0, r1)
                    output.append(r2)
                else:
                    output.append(r1 | r2)
            elif operator == '~':
                output.append(~r2)

        output = []
        stack = []
        precedence = {'|': 1, ' ': 2, '~': 3}
        associativity = {'|': 'left', ' ': 'left', '~': 'right'}
        for token in tokens:
            if isinstance(token, str) and token in (' ', '|', '~'):
                while stack:
                    op = stack[-1]
                    if (op not in ('(', ')') and
                        ((associativity[token] == 'right' and
                          precedence[token] < precedence[op]) or
                         (associativity[token] == 'left' and
                          precedence[token] <= precedence[op]))):
                        apply_operator(output, stack.pop())
                    else:
                        break
                stack.append(token)
            elif isinstance(token, str) and token == '(':
                stack.append(token)
            elif isinstance(token, str) and token == ')':
                while stack[-1] != '(':
                    apply_operator(output, stack.pop())
                    if len(stack) == 0:
                        raise SyntaxError('Mismatched parentheses in '
                                          'region specification.')
                stack.pop()
            else:
                output.append(token)
        while stack:
            if stack[-1] in '()':
                raise SyntaxError('Mismatched parentheses in region '
                                  'specification.')
            apply_operator(output, stack.pop())
        return output[0]

    def clone(self, memo=None):
        """Create a copy of this region - each of the surfaces in the region's
        nodes will be cloned and will have new unique IDs."""
        if memo is None:
            memo = {}
        clone = deepcopy(self)
        clone[:] = [n.clone(memo) for n in self]
        return clone

    def translate(self, vector, inplace=False, memo=None):
        """Translate region in given direction"""
        if memo is None:
            memo = {}
        return type(self)(n.translate(vector, inplace, memo) for n in self)

    def rotate(self, rotation, pivot=(0., 0., 0.), order='xyz', inplace=False,
               memo=None):
        """Rotate surface by angles provided or by applying matrix directly."""
        if memo is None:
            memo = {}
        return type(self)(n.rotate(rotation, pivot=pivot, order=order,
                                   inplace=inplace, memo=memo) for n in self)

    def plot(self, *args, **kwargs):
        """Plotting is not supported by UM2; this is a no-op."""
        from ._stubs import _ignored
        _ignored('Region.plot')


class Intersection(Region, MutableSequence):
    r"""Intersection of two or more regions.

    Instances of Intersection are generally created via the & operator applied
    to two instances of :class:`openmc.Region`.

    Parameters
    ----------
    nodes : iterable of openmc.Region
        Regions to take the intersection of
    """

    def __init__(self, nodes):
        self._nodes = list(nodes)
        for node in self._nodes:
            if not isinstance(node, Region):
                raise ValueError('Intersection operands must be of type Region')

    def __and__(self, other):
        new = Intersection(self)
        new &= other
        return new

    def __iand__(self, other):
        if isinstance(other, Intersection):
            self.extend(other)
        else:
            self.append(other)
        return self

    def __invert__(self) -> Union:
        return Union(~n for n in self)

    def __getitem__(self, key):
        return self._nodes[key]

    def __setitem__(self, key, value):
        self._nodes[key] = value

    def __delitem__(self, key):
        del self._nodes[key]

    def __len__(self):
        return len(self._nodes)

    def insert(self, index, value):
        self._nodes.insert(index, value)

    def __contains__(self, point):
        return all(point in n for n in self)

    def __str__(self):
        return '(' + ' '.join(map(str, self)) + ')'

    @property
    def bounding_box(self) -> BoundingBox:
        box = BoundingBox.infinite()
        for n in self:
            box &= n.bounding_box
        return box


class Union(Region, MutableSequence):
    r"""Union of two or more regions.

    Instances of Union are generally created via the | operator applied to two
    instances of :class:`openmc.Region`.

    Parameters
    ----------
    nodes : iterable of openmc.Region
        Regions to take the union of
    """

    def __init__(self, nodes):
        self._nodes = list(nodes)
        for node in self._nodes:
            if not isinstance(node, Region):
                raise ValueError('Union operands must be of type Region')

    def __or__(self, other):
        new = Union(self)
        new |= other
        return new

    def __ior__(self, other):
        if isinstance(other, Union):
            self.extend(other)
        else:
            self.append(other)
        return self

    def __invert__(self) -> Intersection:
        return Intersection(~n for n in self)

    def __getitem__(self, key):
        return self._nodes[key]

    def __setitem__(self, key, value):
        self._nodes[key] = value

    def __delitem__(self, key):
        del self._nodes[key]

    def __len__(self):
        return len(self._nodes)

    def insert(self, index, value):
        self._nodes.insert(index, value)

    def __contains__(self, point):
        return any(point in n for n in self)

    def __str__(self):
        return '(' + ' | '.join(map(str, self)) + ')'

    @property
    def bounding_box(self) -> BoundingBox:
        bbox = BoundingBox(np.array([np.inf]*3), np.array([-np.inf]*3))
        for n in self:
            bbox |= n.bounding_box
        return bbox


class Complement(Region):
    """Complement of a region.

    The Complement of an existing :class:`openmc.Region` can be created by
    using the ~ operator.

    Parameters
    ----------
    node : openmc.Region
        Region to take the complement of
    """

    def __init__(self, node: Region):
        self.node = node

    def __contains__(self, point):
        return point not in self.node

    def __invert__(self) -> Region:
        return self.node

    def __str__(self):
        return '~' + str(self.node)

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node):
        if not isinstance(node, Region):
            raise ValueError('Complement operand must be of type Region')
        self._node = node

    @property
    def bounding_box(self) -> BoundingBox:
        return (~self.node).bounding_box

    def get_surfaces(self, surfaces=None):
        return self.node.get_surfaces(surfaces)

    def remove_redundant_surfaces(self, redundant_surfaces):
        self.node.remove_redundant_surfaces(redundant_surfaces)

    def clone(self, memo=None):
        if memo is None:
            memo = {}
        clone = deepcopy(self)
        clone.node = self.node.clone(memo)
        return clone

    def translate(self, vector, inplace=False, memo=None):
        if memo is None:
            memo = {}
        return type(self)(self.node.translate(vector, inplace, memo))

    def rotate(self, rotation, pivot=(0., 0., 0.), order='xyz', inplace=False,
               memo=None):
        if memo is None:
            memo = {}
        return type(self)(self.node.rotate(rotation, pivot=pivot, order=order,
                                           inplace=inplace, memo=memo))
