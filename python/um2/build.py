"""Translate OpenMC-style CSG geometry into a 2D gmsh (OpenCASCADE) model.

The geometry described with :mod:`um2.openmc` is built in the current gmsh model
as a set of conformal planar faces. Every face belongs to exactly one physical
group named ``Material_<name>``, which is the convention UM2 uses to associate
mesh elements with materials.

The translation follows the semantics of OpenMC:

* A cell is the intersection of its region with the domain of its universe. The
  domain of the root universe is its bounding box; the domain of a universe
  that fills a cell is the region of that cell; the domain of a universe in a
  lattice element is the element (a rectangle or hexagon), intersected with the
  region of the cell containing the lattice.
* A universe that fills a cell is translated and rotated (about z) as specified
  by the cell.
* The outer universe of a lattice fills the part of the lattice cell that is
  not covered by lattice elements, and is tiled like the lattice itself.

Regions are converted to faces with boolean operations: -cylinder is a disk,
+cylinder is the complement of a disk, the half-space of a plane is a half-box,
& is intersection, | is union and ~ is complement. Each unique (universe,
domain) combination is built once and then copied, so that lattices with many
repeated pins are built efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import warnings

import numpy as np

from .errors import UM2NotSupportedError
from .openmc.cell import Cell
from .openmc.geometry import Geometry
from .openmc.lattice import HexLattice, Lattice, RectLattice
from .openmc.material import Material
from .openmc.region import Complement, Intersection, Region, Union
from .openmc.surface import Halfspace, PlaneMixin, ZCylinder
from .openmc.universe import Universe, UniverseBase

#: Name of the material used for void cells (cells with no fill)
VOID_MATERIAL_NAME = 'Void'

# Relative tolerance used to compare areas and coordinates
_RTOL = 1e-9


def _gmsh():
    import gmsh
    return gmsh


# ==============================================================================
# Materials
# ==============================================================================

def material_names(materials) -> dict:
    """Return a mapping from each material to a unique UM2 material name.

    UM2 identifies materials by the name of a gmsh physical group,
    ``Material_<name>``, which is also used in exported mesh files. Names are
    made of letters, digits, '_', '-' and '.'; other characters are replaced by
    '_'. Materials without a name are named ``Material<id>``, and duplicate
    names are made unique by appending the material ID.

    Parameters
    ----------
    materials : iterable of um2.openmc.Material

    Returns
    -------
    dict
        Mapping from :class:`um2.openmc.Material` to str
    """
    names = {}
    used = set()
    for mat in sorted(set(materials), key=lambda m: m.id):
        name = re.sub(r'[^A-Za-z0-9_.\-]+', '_', mat.name.strip()).strip('_')
        if not name:
            name = f'Material{mat.id}'
        if name in used or name == VOID_MATERIAL_NAME:
            name = f'{name}_{mat.id}'
        used.add(name)
        names[mat] = name
    return names


# ==============================================================================
# Planar geometry helpers
# ==============================================================================

@dataclass(frozen=True)
class Box2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def corners(self):
        return ((self.xmin, self.ymin), (self.xmax, self.ymin),
                (self.xmax, self.ymax), (self.xmin, self.ymax))

    @property
    def size(self):
        return max(self.xmax - self.xmin, self.ymax - self.ymin)

    def expand(self, d):
        return Box2D(self.xmin - d, self.ymin - d, self.xmax + d, self.ymax + d)


def _clip_polygon(points, a, b, d):
    """Clip a convex polygon to the half-plane a*x + b*y <= d (Sutherland-Hodgman)."""
    out = []
    n = len(points)
    for i in range(n):
        p = points[i]
        q = points[(i + 1) % n]
        fp = a*p[0] + b*p[1] - d
        fq = a*q[0] + b*q[1] - d
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((p[0] + t*(q[0] - p[0]), p[1] + t*(q[1] - p[1])))
    return out


def _rotate_point(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c*x - s*y, s*x + c*y


@dataclass
class Transform:
    """A rotation about the z-axis followed by a translation (local -> parent)."""
    angle: float = 0.0          # radians, counterclockwise
    dx: float = 0.0
    dy: float = 0.0

    @property
    def is_identity(self):
        return self.angle == 0.0 and self.dx == 0.0 and self.dy == 0.0

    @classmethod
    def from_cell(cls, cell: Cell):
        angle = 0.0
        if cell.rotation is not None:
            # OpenMC maps parent to local coordinates with local = R (p - t),
            # so local to parent is p = R^T local + t.
            rt = np.asarray(cell.rotation_matrix).T
            angle = math.atan2(rt[1, 0], rt[0, 0])
        dx = dy = 0.0
        if cell.translation is not None:
            dx, dy = float(cell.translation[0]), float(cell.translation[1])
        return cls(angle, dx, dy)

    def apply(self, dimtags):
        """Apply the transform to gmsh entities."""
        occ = _gmsh().model.occ
        if self.angle != 0.0:
            occ.rotate(dimtags, 0, 0, 0, 0, 0, 1, self.angle)
        if self.dx != 0.0 or self.dy != 0.0:
            occ.translate(dimtags, self.dx, self.dy, 0)

    def inverse_point(self, x, y):
        """Map a point in parent coordinates to local coordinates."""
        return _rotate_point(x - self.dx, y - self.dy, -self.angle)

    def inverse_apply(self, dimtags):
        occ = _gmsh().model.occ
        if self.dx != 0.0 or self.dy != 0.0:
            occ.translate(dimtags, -self.dx, -self.dy, 0)
        if self.angle != 0.0:
            occ.rotate(dimtags, 0, 0, 0, 0, 0, 1, -self.angle)


# ==============================================================================
# Axial (z) bounds
# ==============================================================================

@dataclass
class AxialBounds:
    """Axial extent of the model, from z-planes bounding the root cells."""
    zmin: float = -math.inf
    zmax: float = math.inf
    boundary_types: tuple = ()

    @property
    def is_bounded(self):
        return math.isfinite(self.zmin) and math.isfinite(self.zmax)


def _axial_interval(region):
    """Return the (zmin, zmax, [boundary types]) interval imposed by the
    axial-plane half-spaces of a region, or None if it has none. Raises if an
    axial plane is used in a way that is not an axial bound of the whole cell."""
    if region is None:
        return None

    def contains_axial(node):
        if isinstance(node, Halfspace):
            return isinstance(node.surface, PlaneMixin) and node.surface.is_axial
        if isinstance(node, Complement):
            return contains_axial(node.node)
        return any(contains_axial(n) for n in node)

    zmin, zmax, bcs = -math.inf, math.inf, []
    found = False
    nodes = list(region) if isinstance(region, Intersection) else [region]
    for node in nodes:
        if isinstance(node, Halfspace) and isinstance(node.surface, PlaneMixin) \
                and node.surface.is_axial:
            surf = node.surface
            z0 = surf.d / surf.c
            # c*z - d < 0 is '-'. For c > 0 that is z < z0.
            upper = (node.side == '-') == (surf.c > 0)
            if upper:
                zmax = min(zmax, z0)
            else:
                zmin = max(zmin, z0)
            bcs.append(surf.boundary_type)
            found = True
        elif isinstance(node, Intersection):
            sub = _axial_interval(node)
            if sub is not None:
                zmin, zmax = max(zmin, sub[0]), min(zmax, sub[1])
                bcs.extend(sub[2])
                found = True
        elif contains_axial(node):
            raise UM2NotSupportedError(
                'z-planes may only be used to bound cells axially (intersected '
                'with the rest of the region); they cannot appear inside a union '
                'or complement, since UM2 geometry must be Z-invariant.')
    return (zmin, zmax, bcs) if found else None


# ==============================================================================
# Builder
# ==============================================================================

@dataclass
class BuildResult:
    """Result of :func:`build_geometry`."""
    #: Mapping from UM2 material name to the um2.openmc.Material (None for void)
    materials: dict = field(default_factory=dict)
    #: Mapping from UM2 material name to a list of gmsh surface tags
    material_surfaces: dict = field(default_factory=dict)
    #: Bounding box (xmin, ymin, xmax, ymax) of the model
    bounding_box: tuple = (0., 0., 0., 0.)
    #: Axial bounds of the model
    axial_bounds: AxialBounds = field(default_factory=AxialBounds)
    #: Boundary conditions of the surfaces that bound the root universe
    boundary_types: set = field(default_factory=set)

    def material_areas(self) -> dict:
        """Return the total area of each material, as built. The surface tags
        are only valid until the gmsh model is modified (e.g. by the coarse
        grid overlay); use :func:`material_areas` afterwards."""
        occ = _gmsh().model.occ
        return {name: sum(occ.getMass(2, t) for t in tags)
                for name, tags in self.material_surfaces.items()}


def material_areas(dim=2) -> dict:
    """Return the total area of each ``Material_<name>`` physical group of the
    current gmsh model, keyed by material name."""
    gmsh = _gmsh()
    areas = {}
    for d, tag in gmsh.model.getPhysicalGroups(dim):
        name = gmsh.model.getPhysicalName(d, tag)
        if name.startswith('Material_'):
            ents = gmsh.model.getEntitiesForPhysicalGroup(d, tag)
            areas[name[len('Material_'):]] = sum(gmsh.model.occ.getMass(d, int(e))
                                                 for e in ents)
    return areas


class _Builder:
    """Builds the faces of a geometry in the gmsh OCC kernel."""

    def __init__(self, names, void_name):
        self.names = names
        self.void_name = void_name
        # Template cache: key -> list of (tag, material name) at the origin
        self.templates = {}
        self.template_tags = []
        # Faces to remove. Removing entities from the OCC kernel is expensive
        # (it scales with the size of the model), so removal is batched.
        self.garbage = []
        self.void_used = False

    # --------------------------------------------------------------------------
    # OCC helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def occ():
        return _gmsh().model.occ

    def bbox(self, tags):
        """2D bounding box of a set of surface tags."""
        xmin = ymin = math.inf
        xmax = ymax = -math.inf
        for t in tags:
            x0, y0, _, x1, y1, _ = self.occ().getBoundingBox(2, t)
            xmin, ymin = min(xmin, x0), min(ymin, y0)
            xmax, ymax = max(xmax, x1), max(ymax, y1)
        return Box2D(xmin, ymin, xmax, ymax)

    def area(self, tags):
        return sum(self.occ().getMass(2, t) for t in tags)

    def copy(self, tags):
        if not tags:
            return []
        return [t for _, t in self.occ().copy([(2, t) for t in tags])]

    def remove(self, tags):
        self.garbage.extend(tags)

    def collect_garbage(self):
        # Non-recursive removal is fast; the curves and points of the removed
        # faces are removed afterwards by _remove_orphans.
        if self.garbage:
            self.occ().remove([(2, t) for t in self.garbage], recursive=False)
            self.garbage = []

    def polygon(self, points):
        """Create a planar face from a polygon."""
        occ = self.occ()
        pts = [occ.addPoint(x, y, 0) for x, y in points]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        loop = occ.addCurveLoop(lines)
        return occ.addPlaneSurface([loop])

    def boolean(self, op, tags, tool_tags):
        """Apply a boolean operation, consuming both inputs."""
        if not tags:
            self.remove(tool_tags)
            return []
        if not tool_tags:
            if op == 'intersect':
                self.remove(tags)
                return []
            return tags
        func = getattr(self.occ(), op)
        out, _ = func([(2, t) for t in tags], [(2, t) for t in tool_tags])
        return [t for d, t in out if d == 2]

    # --------------------------------------------------------------------------
    # Regions
    # --------------------------------------------------------------------------

    def apply_region(self, node: Region, tags):
        """Intersect faces with a region. Consumes tags; returns new tags."""
        if not tags:
            return []
        if isinstance(node, Halfspace):
            return self.apply_halfspace(node, tags)
        if isinstance(node, Intersection):
            for child in node:
                tags = self.apply_region(child, tags)
                if not tags:
                    break
            return tags
        if isinstance(node, Union):
            pieces = []
            for child in node:
                pieces.extend(self.apply_region(child, self.copy(tags)))
            self.remove(tags)
            if len(pieces) <= 1:
                return pieces
            out, _ = self.occ().fuse([(2, pieces[0])], [(2, t) for t in pieces[1:]])
            return [t for d, t in out if d == 2]
        if isinstance(node, Complement):
            inner = self.apply_region(node.node, self.copy(tags))
            return self.boolean('cut', tags, inner)
        raise TypeError(f'Unknown region type {type(node).__name__}')

    def apply_halfspace(self, node: Halfspace, tags):
        surf = node.surface
        box = self.bbox(tags)
        if isinstance(surf, ZCylinder):
            x0, y0, r = surf.x0, surf.y0, surf.r
            # Closest and farthest distance from the center to the box
            dx = max(box.xmin - x0, 0.0, x0 - box.xmax)
            dy = max(box.ymin - y0, 0.0, y0 - box.ymax)
            nearest = math.hypot(dx, dy)
            farthest = max(math.hypot(cx - x0, cy - y0) for cx, cy in box.corners)
            tol = _RTOL * max(r, box.size)
            if nearest >= r - tol:          # disk does not overlap the box
                if node.side == '+':
                    return tags
                self.remove(tags)
                return []
            if farthest <= r + tol:         # box is inside the disk
                if node.side == '-':
                    return tags
                self.remove(tags)
                return []
            disk = self.occ().addDisk(x0, y0, 0, r, r)
            op = 'intersect' if node.side == '-' else 'cut'
            return self.boolean(op, tags, [disk])

        if isinstance(surf, PlaneMixin):
            if surf.is_axial:
                # Axial bounds are validated separately and do not affect the
                # 2D geometry.
                return tags
            a, b, _, d = surf._get_base_coeffs()
            if node.side == '+':
                a, b, d = -a, -b, -d
            # Now the region is a*x + b*y - d <= 0
            vals = [a*x + b*y - d for x, y in box.corners]
            tol = _RTOL * math.hypot(a, b) * max(box.size, 1.0)
            if max(vals) <= tol:
                return tags
            if min(vals) >= -tol:
                self.remove(tags)
                return []
            big = box.expand(max(box.size, 1.0))
            poly = _clip_polygon(list(big.corners), a, b, d)
            return self.boolean('intersect', tags, [self.polygon(poly)])

        raise UM2NotSupportedError(f'Surfaces of type {type(surf).__name__} are not '
                                   'supported by UM2')

    # --------------------------------------------------------------------------
    # Universes
    # --------------------------------------------------------------------------

    def material_name(self, cell: Cell):
        if cell.fill_type == 'material':
            return self.names[cell.fill]
        self.void_used = True
        return self.void_name

    def build_universe(self, universe: UniverseBase, domain_tags):
        """Build the cells of a universe clipped to a domain (in the universe's
        local coordinates). Consumes domain_tags. Returns (tag, material) pairs."""
        if not isinstance(universe, Universe):
            raise UM2NotSupportedError(
                f'{type(universe).__name__} universes are not supported by UM2')
        result = []
        cells = list(universe.cells.values())
        for i, cell in enumerate(cells):
            last = i == len(cells) - 1
            tags = domain_tags if last else self.copy(domain_tags)
            if cell.region is not None:
                tags = self.apply_region(cell.region, tags)
            if not tags:
                continue
            result.extend(self.build_fill(cell, tags))
        if not cells:
            self.remove(domain_tags)
        return result

    def build_fill(self, cell: Cell, tags):
        """Fill faces (in the cell's coordinates) with the contents of a cell.
        Consumes tags."""
        if cell.fill_type in ('material', 'void'):
            name = self.material_name(cell)
            return [(t, name) for t in tags]

        transform = Transform.from_cell(cell)
        # Map the domain into the fill's local coordinates
        transform.inverse_apply([(2, t) for t in tags])
        if cell.fill_type == 'universe':
            pieces = self.build_universe(cell.fill, tags)
        else:
            pieces = self.build_lattice(cell.fill, tags)
        transform.apply([(2, t) for t, _ in pieces])
        return pieces

    # --------------------------------------------------------------------------
    # Lattices
    # --------------------------------------------------------------------------

    def element_polygon(self, lattice: Lattice, center=(0.0, 0.0)):
        """Vertices of a lattice element centered at `center`."""
        cx, cy = center
        if isinstance(lattice, RectLattice):
            hx, hy = lattice.pitch[0] / 2, lattice.pitch[1] / 2
            return [(cx - hx, cy - hy), (cx + hx, cy - hy),
                    (cx + hx, cy + hy), (cx - hx, cy + hy)]
        # Hexagonal element (Voronoi cell of the lattice centers)
        R = lattice.pitch[0] / math.sqrt(3.0)
        offset = 0.0 if lattice.orientation == 'y' else math.pi / 6
        return [(cx + R*math.cos(offset + k*math.pi/3),
                 cy + R*math.sin(offset + k*math.pi/3)) for k in range(6)]

    def lattice_elements(self, lattice: Lattice, box: Box2D):
        """Yield (natural index, center, universe, is_valid) for every element
        of the (infinitely tiled) lattice whose element overlaps the box."""
        if isinstance(lattice, RectLattice):
            px, py = lattice.pitch[0], lattice.pitch[1]
            llx, lly = lattice.lower_left[0], lattice.lower_left[1]
            ix0 = math.floor((box.xmin - llx) / px - _RTOL)
            ix1 = math.floor((box.xmax - llx) / px + _RTOL)
            iy0 = math.floor((box.ymin - lly) / py - _RTOL)
            iy1 = math.floor((box.ymax - lly) / py + _RTOL)
            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    idx = (ix, iy)
                    valid = lattice.is_valid_index(idx)
                    yield idx, lattice.element_center(idx), valid
        else:
            p = lattice.pitch[0]
            cx, cy = lattice.center[0], lattice.center[1]
            # Number of rings needed to cover the box
            far = max(math.hypot(x - cx, y - cy) for x, y in box.corners)
            n = int(math.ceil(far / (p * math.sqrt(0.75)))) + 2
            n = max(n, lattice.num_rings)
            R = p / math.sqrt(3.0)
            for a in range(-n + 1, n):
                for x in range(-n + 1, n):
                    if max(abs(x), abs(a), abs(x + a)) >= n:
                        continue
                    idx = (x, a)
                    ex, ey = lattice.element_center(idx)
                    if (ex + R < box.xmin or ex - R > box.xmax or
                            ey + R < box.ymin or ey - R > box.ymax):
                        continue
                    yield idx, (ex, ey), lattice.is_valid_index(idx)

    def build_lattice(self, lattice: Lattice, domain_tags):
        """Build a lattice clipped to a domain. Consumes domain_tags."""
        if lattice.universes is None:
            raise ValueError(f'Lattice {lattice.id} does not have universes assigned.')
        if isinstance(lattice, RectLattice) and lattice.lower_left is None:
            raise ValueError(f'Lattice {lattice.id} does not have lower_left specified.')
        if isinstance(lattice, HexLattice) and lattice.center is None:
            raise ValueError(f'Lattice {lattice.id} does not have center specified.')

        box = self.bbox(domain_tags)
        domain_area = self.area(domain_tags)
        tol = _RTOL * max(box.size, 1.0)**2
        # If the domain is a rectangle, elements can be classified exactly by
        # comparing bounding boxes instead of with boolean operations.
        box_area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
        # (OCC bounding boxes are enlarged by a small tolerance.)
        eps = 1e-6 * max(box.size, 1.0)
        rectangular = (len(domain_tags) == 1 and
                       abs(box_area - domain_area) <= 4 * eps * box.size)
        result = []
        for idx, center, valid in list(self.lattice_elements(lattice, box)):
            if valid:
                universe = lattice.get_universe(idx)
            elif lattice.outer is not None:
                universe = lattice.outer
            else:
                continue
            vertices = self.element_polygon(lattice, center)
            if rectangular:
                exmin = min(x for x, _ in vertices)
                exmax = max(x for x, _ in vertices)
                eymin = min(y for _, y in vertices)
                eymax = max(y for _, y in vertices)
                if (exmax <= box.xmin + eps or exmin >= box.xmax - eps or
                        eymax <= box.ymin + eps or eymin >= box.ymax - eps):
                    continue
                inside = (exmin >= box.xmin - eps and exmax <= box.xmax + eps and
                          eymin >= box.ymin - eps and eymax <= box.ymax + eps)
                clipped = None
            else:
                element = self.polygon(vertices)
                element_area = self.area([element])
                clipped = self.boolean('intersect', [element], self.copy(domain_tags))
                if not clipped:
                    continue
                clipped_area = self.area(clipped)
                if clipped_area <= tol:
                    self.remove(clipped)
                    continue
                inside = abs(clipped_area - element_area) <= tol * 10
                if inside:
                    self.remove(clipped)
            if not inside and clipped is None:
                clipped = self.boolean('intersect', [self.polygon(vertices)],
                                       self.copy(domain_tags))
                if not clipped:
                    continue
            if inside:
                # The whole element is in the domain: use (or create) a template
                key = (universe.id, type(lattice).__name__, tuple(lattice.pitch),
                       getattr(lattice, 'orientation', None))
                pieces = self.instance(key, universe, lattice)
                self.occ().translate([(2, t) for t, _ in pieces], center[0], center[1], 0)
            else:
                # Partial element: build it uniquely in element coordinates
                self.occ().translate([(2, t) for t in clipped], -center[0], -center[1], 0)
                pieces = self.build_universe(universe, clipped)
                self.occ().translate([(2, t) for t, _ in pieces], center[0], center[1], 0)
            result.extend(pieces)
        covered = sum(self.area([t]) for t, _ in result)
        if covered < domain_area - 1e-6 * max(domain_area, 1.0):
            warnings.warn(
                f'Lattice {lattice.id} does not cover the region of the cell that it '
                f'fills (uncovered area {domain_area - covered:.6g} cm^2). Set the '
                'lattice outer universe to fill the remainder.')
        self.remove(domain_tags)
        return result

    def instance(self, key, universe, lattice):
        """Return a copy of the template for a universe in a whole lattice
        element (centered at the origin), creating the template if needed."""
        if key not in self.templates:
            element = self.polygon(self.element_polygon(lattice))
            pieces = self.build_universe(universe, [element])
            self.templates[key] = pieces
            self.template_tags.extend(t for t, _ in pieces)
        pieces = self.templates[key]
        tags = self.copy([t for t, _ in pieces])
        return [(t, name) for t, (_, name) in zip(tags, pieces)]

    def remove_templates(self):
        self.remove(self.template_tags)
        self.template_tags = []
        self.templates = {}


def _polygon_box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Box2D(min(xs), min(ys), max(xs), max(ys))


def _region_polygon(node, polygon):
    """Return a convex polygon that contains the part of `polygon` inside the
    region (exact for intersections of plane half-spaces; conservative
    otherwise). Returns an empty list if the region does not intersect it."""
    if not polygon:
        return polygon
    if isinstance(node, Halfspace):
        surf = node.surface
        if isinstance(surf, PlaneMixin):
            if surf.is_axial:
                return polygon
            a, b, _, d = surf._get_base_coeffs()
            if node.side == '+':
                a, b, d = -a, -b, -d
            return _clip_polygon(polygon, a, b, d)
        if isinstance(surf, ZCylinder) and node.side == '-':
            x0, y0, r = surf.x0, surf.y0, surf.r
            for a, b, d in ((1, 0, x0 + r), (-1, 0, r - x0), (0, 1, y0 + r),
                            (0, -1, r - y0)):
                polygon = _clip_polygon(polygon, a, b, d)
            return polygon
        return polygon
    if isinstance(node, Intersection):
        for child in node:
            polygon = _region_polygon(child, polygon)
        return polygon
    if isinstance(node, Union):
        pieces = [p for p in (_region_polygon(c, polygon) for c in node) if p]
        if not pieces:
            return []
        return list(_polygon_box([pt for p in pieces for pt in p]).corners)
    return polygon   # Complement: conservative


def _root_box(geometry: Geometry) -> Box2D:
    """Bounding box (in x and y) of the root universe."""
    bb = geometry.bounding_box
    finite = [v for v in (bb.lower_left[0], bb.lower_left[1], bb.upper_right[0],
                          bb.upper_right[1]) if math.isfinite(v)]
    big = 1e3 * max([abs(v) for v in finite] + [1e3])
    start = Box2D(*(v if math.isfinite(v) else s*big for v, s in
                    ((bb.lower_left[0], -1), (bb.lower_left[1], -1),
                     (bb.upper_right[0], 1), (bb.upper_right[1], 1))))
    boxes = []
    for cell in geometry.root_universe.cells.values():
        if cell.region is None:
            boxes = None
            break
        polygon = _region_polygon(cell.region, list(start.corners))
        if polygon:
            boxes.append(_polygon_box(polygon))
    box = None
    if boxes:
        box = Box2D(min(b.xmin for b in boxes), min(b.ymin for b in boxes),
                    max(b.xmax for b in boxes), max(b.ymax for b in boxes))
    if box is None or max(abs(v) for v in (box.xmin, box.ymin, box.xmax, box.ymax)) \
            >= big * (1 - 1e-9):
        raise ValueError(
            'The root universe must be bounded in x and y (e.g. by a '
            'RectangularPrism or a ZCylinder with a vacuum/reflective boundary), '
            f'but its bounding box is {bb}.')
    return box


def _check_axial(geometry: Geometry) -> AxialBounds:
    """Check that z-planes are only used to bound the model axially and return
    the axial bounds."""
    intervals = []
    for cell in geometry.get_all_cells().values():
        interval = _axial_interval(cell.region)
        if interval is not None:
            intervals.append((cell, interval))
    bounds = AxialBounds()
    if not intervals:
        return bounds
    z0, z1 = intervals[0][1][0], intervals[0][1][1]
    for cell, (zmin, zmax, _) in intervals:
        if not (math.isclose(zmin, z0, abs_tol=1e-9) and
                math.isclose(zmax, z1, abs_tol=1e-9)):
            raise UM2NotSupportedError(
                f'Cell {cell.id} is bounded axially by z in [{zmin}, {zmax}], but '
                f'other cells are bounded by z in [{z0}, {z1}]. UM2 geometry must '
                'be Z-invariant: z-planes may only bound the whole model axially.')
    bcs = tuple(sorted({bc for _, (_, _, b) in intervals for bc in b}))
    bounds = AxialBounds(z0, z1, bcs)
    if 'vacuum' in bcs:
        warnings.warn(
            'The model has a vacuum boundary in z, which is ignored by UM2 since '
            'the geometry is 2D (Z-invariant). The resulting 2D model has no axial '
            'leakage.')
    return bounds


def _boundary_types(geometry: Geometry):
    types = set()
    for cell in geometry.root_universe.cells.values():
        if cell.region is not None:
            for surf in cell.region.get_surfaces().values():
                if surf.boundary_type != 'transmission':
                    types.add(surf.boundary_type)
    return types


def _remove_orphans():
    """Remove curves and points that do not bound any surface."""
    gmsh = _gmsh()
    occ = gmsh.model.occ
    occ.synchronize()
    faces = gmsh.model.getEntities(2)
    used_curves = set(gmsh.model.getBoundary(faces, combined=False, oriented=False,
                                             recursive=False)) if faces else set()
    orphan_curves = [dt for dt in gmsh.model.getEntities(1) if dt not in used_curves]
    if orphan_curves:
        occ.remove(orphan_curves, recursive=False)
    used_points = set(gmsh.model.getBoundary(faces, combined=False, oriented=False,
                                             recursive=True)) if faces else set()
    orphan_points = [dt for dt in gmsh.model.getEntities(0) if dt not in used_points]
    if orphan_points:
        occ.remove(orphan_points, recursive=False)
    occ.synchronize()


def build_geometry(geometry, *, void_material_name=VOID_MATERIAL_NAME,
                   conformal=True, clip_boxes=None, offset=(0.0, 0.0)) -> BuildResult:
    """Build an OpenMC-style geometry in the current gmsh model.

    gmsh must be initialized (e.g. with :func:`um2.initialize`) before calling
    this function. The geometry is added to the current gmsh model as 2D faces
    in the plane z = 0, with one physical group ``Material_<name>`` per
    material. Entity colors are set from the material colors.

    Parameters
    ----------
    geometry : um2.openmc.Geometry or um2.openmc.Model
        Geometry to build
    void_material_name : str
        Name of the material assigned to void cells (cells without a fill)
    conformal : bool
        Whether to fragment the faces so that neighboring faces share their
        boundaries (a conformal model). This is required to mesh the model
        directly, but not when the model is subsequently fragmented anyway,
        e.g. by UM2's coarse grid overlay. Fragmenting large models is
        expensive.
    clip_boxes : list of tuple, optional
        If given, only the parts of the geometry inside these boxes
        ``(xmin, ymin, xmax, ymax)`` are built. The boxes must not overlap.
        This is used to build only the unique coarse cells of an MPACT model.
    offset : tuple of float
        Translation (dx, dy) applied to the built geometry. UM2's MPACT models
        have their lower-left corner at the origin.

    Returns
    -------
    BuildResult
        The materials, their surfaces, and the bounds of the model
    """
    from .openmc.model import Model
    if isinstance(geometry, Model):
        geometry = geometry.geometry
    if not isinstance(geometry, Geometry):
        raise TypeError('build_geometry expects a um2.openmc.Geometry or Model')
    if geometry.root_universe is None:
        raise ValueError('The geometry has no root universe')

    gmsh = _gmsh()
    if not gmsh.isInitialized():
        raise RuntimeError('gmsh is not initialized; call um2.initialize() first')

    axial = _check_axial(geometry)
    box = _root_box(geometry)
    materials = list(geometry.get_all_materials().values())
    names = material_names(materials)
    if len(names) + 1 > 127:
        raise ValueError('UM2 supports at most 127 materials')

    builder = _Builder(names, void_material_name)
    if clip_boxes is None:
        domains = [box]
    else:
        domains = [Box2D(*b) for b in clip_boxes]
    faces = []
    for domain in domains:
        root = builder.polygon(list(domain.corners))
        faces.extend(builder.build_universe(geometry.root_universe, [root]))
    builder.remove_templates()
    builder.collect_garbage()
    dx, dy = offset
    if faces and (dx != 0.0 or dy != 0.0):
        gmsh.model.occ.translate([(2, t) for t, _ in faces], dx, dy, 0)
        box = Box2D(box.xmin + dx, box.ymin + dy, box.xmax + dx, box.ymax + dy)

    # Make the faces conformal. If faces overlap (which is an error in OpenMC),
    # the first one wins.
    occ = gmsh.model.occ
    result = BuildResult(bounding_box=(box.xmin, box.ymin, box.xmax, box.ymax),
                         axial_bounds=axial,
                         boundary_types=_boundary_types(geometry))
    owner = {}
    if faces and not conformal:
        for tag, name in faces:
            owner[tag] = name
    elif faces:
        dimtags = [(2, t) for t, _ in faces]
        _, out_map = occ.fragment(dimtags, [])
        for (_, name), pieces in zip(faces, out_map):
            for dim, tag in pieces:
                if dim == 2 and tag not in owner:
                    owner[tag] = name
    for surface_name in sorted(set(owner.values())):
        result.material_surfaces[surface_name] = sorted(
            t for t, n in owner.items() if n == surface_name)
    for mat, name in names.items():
        if name in result.material_surfaces:
            result.materials[name] = mat
    if builder.void_used and void_material_name in result.material_surfaces:
        result.materials[void_material_name] = None

    _remove_orphans()
    from .colors import material_color
    for name, tags in result.material_surfaces.items():
        gmsh.model.addPhysicalGroup(2, tags, name=f'Material_{name}')
        r, g, b = material_color(name)
        gmsh.model.setColor([(2, t) for t in tags], r, g, b, recursive=True)
    return result
