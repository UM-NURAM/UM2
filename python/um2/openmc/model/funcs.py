"""Helper functions (subset of :mod:`openmc.model.funcs`)."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from operator import attrgetter
from warnings import warn

from ..cell import Cell
from ..checkvalue import check_iterable_type, check_length, check_less_than, check_type
from ..surface import ZCylinder
from ..universe import Universe
from .surface_composite import HexagonalPrism, RectangularPrism


def rectangular_prism(width, height, axis='z', origin=(0., 0.),
                      boundary_type='transmission', corner_radius=0.):
    """Get an infinite rectangular prism from four planar surfaces.

    .. deprecated:: Use :class:`openmc.model.RectangularPrism` instead.
    """
    warn("The rectangular_prism(...) function has been replaced by the "
         "RectangularPrism(...) class. Future versions of OpenMC will not "
         "accept rectangular_prism.", FutureWarning)
    return -RectangularPrism(width=width, height=height, axis=axis, origin=origin,
                             boundary_type=boundary_type, corner_radius=corner_radius)


def hexagonal_prism(edge_length=1., orientation='y', origin=(0., 0.),
                    boundary_type='transmission', corner_radius=0.):
    """Create a hexagon region from six surface planes.

    .. deprecated:: Use :class:`openmc.model.HexagonalPrism` instead.
    """
    warn("The hexagonal_prism(...) function has been replaced by the "
         "HexagonalPrism(...) class. Future versions of OpenMC will not "
         "accept hexagonal_prism.", FutureWarning)
    return -HexagonalPrism(edge_length=edge_length, orientation=orientation,
                           origin=origin, boundary_type=boundary_type,
                           corner_radius=corner_radius)


def subdivide(surfaces):
    """Create regions separated by a series of surfaces.

    Parameters
    ----------
    surfaces : sequence of openmc.Surface
        Surfaces separating regions

    Returns
    -------
    list of openmc.Region
        Regions formed by the given surfaces
    """
    regions = [-surfaces[0]]
    for s0, s1 in zip(surfaces[:-1], surfaces[1:]):
        regions.append(+s0 & -s1)
    regions.append(+surfaces[-1])
    return regions


def pin(surfaces, items, subdivisions=None, divide_vols=True, **kwargs):
    """Convenience function for building a fuel pin

    Parameters
    ----------
    surfaces : iterable of openmc.ZCylinder
        Cylinders used to define boundaries between items. All cylinders must
        be concentric and increasing in radius.
    items : iterable
        Objects to go between ``surfaces``. There must be one more item than
        surfaces.
    subdivisions : dict of {int: int}
        Dictionary describing which rings to subdivide and how many times.
        Keys are indexes of the annular rings to be divided. Values are the
        number of divisions. The new rings have equal volume.
    divide_vols : bool
        If this evaluates to ``True``, then volumes of subdivided materials
        will also be divided by the number of divisions.
    **kwargs
        Additional key-word arguments to be passed to :class:`openmc.Universe`

    Returns
    -------
    openmc.Universe
        Universe of concentric cylinders filled with the desired items

    Notes
    -----
    Unlike OpenMC, the cylinders created for subdivided rings share the center
    of the input cylinders.
    """
    if "cells" in kwargs:
        raise ValueError("Cells will be set by this function, not from input "
                         "arguments.")
    check_type("items", items, Iterable)
    check_length("surfaces", surfaces, len(items) - 1, len(items) - 1)
    check_type("surface", surfaces[0], ZCylinder)
    check_iterable_type("surfaces", surfaces[1:], ZCylinder)

    center_getter = attrgetter("x0", "y0")
    centers = set()
    prev_rad = 0
    for ix, surf in enumerate(surfaces):
        cur_rad = surf.r
        if cur_rad <= prev_rad:
            raise ValueError(
                "Surfaces do not appear to be increasing in radius. Surface "
                f"{surf.id} at index {ix} has radius {cur_rad:7.3e} compared to "
                f"previous radius of {prev_rad:7.5e}")
        prev_rad = cur_rad
        centers.add(center_getter(surf))
    if len(centers) > 1:
        raise ValueError("Surfaces do not appear to be concentric. The following "
                         f"centers were found: {centers}")
    x0, y0 = centers.pop()

    if subdivisions is not None:
        check_length("subdivisions", subdivisions, 1, len(surfaces))
        orig_indexes = list(subdivisions.keys())
        check_iterable_type("ring indexes", orig_indexes, int)
        check_iterable_type("number of divisions", list(subdivisions.values()), int)
        for ix in orig_indexes:
            if ix < 0:
                subdivisions[len(surfaces) + ix] = subdivisions.pop(ix)
        check_less_than("outer ring", max(subdivisions), len(surfaces), equality=True)

        items = list(items)
        surfaces = list(surfaces)
        for ring_index in reversed(sorted(subdivisions.keys())):
            nr = subdivisions[ring_index]
            new_surfs = []
            lower_rad = 0.0 if ring_index == 0 else surfaces[ring_index - 1].r
            upper_rad = surfaces[ring_index].r
            area_term = (upper_rad ** 2 - lower_rad ** 2) / nr
            for _ in range(nr - 1):
                lower_rad = sqrt(area_term + lower_rad ** 2)
                new_surfs.append(ZCylinder(x0=x0, y0=y0, r=lower_rad))
            surfaces = surfaces[:ring_index] + new_surfs + surfaces[ring_index:]
            filler = items[ring_index]
            if (divide_vols and hasattr(filler, "volume")
                    and filler.volume is not None):
                filler.volume /= nr
            items[ring_index:ring_index] = [filler.clone() for _ in range(nr - 1)]

    regions = subdivide(surfaces)
    cells = [Cell(fill=f, region=r) for r, f in zip(regions, items)]
    return Universe(cells=cells, **kwargs)
