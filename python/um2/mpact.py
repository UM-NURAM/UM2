"""The MPACT spatial partition (core, assemblies, lattices, RTMs, coarse cells).

MPACT models are partitioned hierarchically:

* the **core** is a rectilinear grid of assemblies,
* an **assembly** is an axial stack of lattices (a single lattice in 2D),
* a **lattice** is a regular grid of ray tracing modules (RTMs), which must all
  have the same size,
* an **RTM** is a rectilinear grid of coarse cells, and
* a **coarse cell** is a rectangle that is meshed. Coarse cells that appear
  more than once share a single mesh.

A :class:`Plan` describes this partition for an :mod:`um2.openmc` geometry. It
can be inferred automatically with :func:`infer_hierarchy`, inspected with
``print(plan)``, edited, or built by hand with the ``add_*`` methods, and is
then turned into a UM2 MPACT model by :func:`um2.mesh_geometry`.

Arrays of IDs (``Plan.rtms``, ``Plan.lattices`` and ``Plan.core``) follow the
OpenMC convention for lattices: row 0 is the top (largest y) row.

Coordinates of an MPACT model start at the lower-left corner of the core.
:attr:`Plan.origin` is the location of that corner in the coordinates of the
OpenMC geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings

import numpy as np

from .errors import UM2NotSupportedError
from .openmc.cell import Cell
from .openmc.geometry import Geometry
from .openmc.lattice import HexLattice, RectLattice
from .openmc.surface import Halfspace, XPlane, YPlane
from .openmc.universe import Universe

# Relative tolerance for comparing lengths
_RTOL = 1e-6


def _close(a, b, scale=1.0):
    return abs(a - b) <= _RTOL * max(abs(a), abs(b), scale)


@dataclass
class CoarseCell:
    """A coarse cell of an MPACT model.

    Attributes
    ----------
    width, height : float
        Size of the coarse cell [cm]
    label : str
        Description of the contents of the coarse cell
    key : hashable
        Identity of the contents. Coarse cells are only shared (meshed once)
        when they are the same object in the plan.
    """
    width: float
    height: float
    label: str = ''
    key: object = None


def _as_id_array(ids, name):
    arr = np.array(ids, dtype=int)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f'{name} must be a non-empty 2D array of IDs')
    return arr


@dataclass
class Plan:
    """A (2D) MPACT spatial partition of an OpenMC geometry.

    Build a plan with :func:`infer_hierarchy`, or by hand::

        plan = Plan()
        cc = plan.add_coarse_cell(1.26, 1.26, 'pin')
        rtm = plan.add_rtm([[cc]])
        lat = plan.add_lattice([[rtm] * 17] * 17)
        asy = plan.add_assembly([lat])
        plan.set_core([[asy]])
        plan.origin = (-10.71, -10.71)
    """
    coarse_cells: list = field(default_factory=list)
    rtms: list = field(default_factory=list)
    lattices: list = field(default_factory=list)
    assemblies: list = field(default_factory=list)
    core: np.ndarray | None = None
    #: Lower-left corner of the core in the coordinates of the OpenMC geometry
    origin: tuple = (0.0, 0.0)
    #: Axial extent of the OpenMC model (informational; the UM2 model is 2D)
    z: tuple = (-0.5, 0.5)
    #: Notes recorded during inference (shown by print)
    notes: list = field(default_factory=list)

    # --------------------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------------------

    def add_coarse_cell(self, width, height, label='', key=None) -> int:
        """Add a coarse cell and return its ID."""
        if width <= 0 or height <= 0:
            raise ValueError('Coarse cell dimensions must be positive')
        self.coarse_cells.append(CoarseCell(float(width), float(height), label, key))
        return len(self.coarse_cells) - 1

    def add_rtm(self, coarse_cell_ids) -> int:
        """Add a ray tracing module (a rectilinear grid of coarse cells, rows
        from top to bottom) and return its ID."""
        ids = _as_id_array(coarse_cell_ids, 'coarse_cell_ids')
        for i in ids.flat:
            if not 0 <= i < len(self.coarse_cells):
                raise ValueError(f'Coarse cell {i} does not exist')
        self.rtms.append(ids)
        return len(self.rtms) - 1

    def add_lattice(self, rtm_ids) -> int:
        """Add a lattice (a regular grid of RTMs, rows from top to bottom) and
        return its ID."""
        ids = _as_id_array(rtm_ids, 'rtm_ids')
        for i in ids.flat:
            if not 0 <= i < len(self.rtms):
                raise ValueError(f'RTM {i} does not exist')
        self.lattices.append(ids)
        return len(self.lattices) - 1

    def add_assembly(self, lattice_ids) -> int:
        """Add an assembly (lattices from bottom to top; a single lattice in 2D)
        and return its ID."""
        ids = [int(i) for i in lattice_ids]
        if len(ids) != 1:
            raise UM2NotSupportedError('Only 2D assemblies (a single lattice) are '
                                       'supported')
        for i in ids:
            if not 0 <= i < len(self.lattices):
                raise ValueError(f'Lattice {i} does not exist')
        self.assemblies.append(ids)
        return len(self.assemblies) - 1

    def set_core(self, assembly_ids):
        """Set the core (a rectilinear grid of assemblies, rows from top to
        bottom)."""
        ids = _as_id_array(assembly_ids, 'assembly_ids')
        for i in ids.flat:
            if not 0 <= i < len(self.assemblies):
                raise ValueError(f'Assembly {i} does not exist')
        self.core = ids

    # --------------------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------------------

    def rtm_size(self, rtm_id):
        """(width, height) of an RTM."""
        ids = self.rtms[rtm_id]
        width = sum(self.coarse_cells[i].width for i in ids[0])
        height = sum(self.coarse_cells[i].height for i in ids[:, 0])
        return width, height

    def lattice_size(self, lattice_id):
        ids = self.lattices[lattice_id]
        w, h = self.rtm_size(ids[0, 0])
        return w * ids.shape[1], h * ids.shape[0]

    def assembly_size(self, assembly_id):
        return self.lattice_size(self.assemblies[assembly_id][0])

    @property
    def size(self):
        """(width, height) of the core."""
        if self.core is None:
            return (0.0, 0.0)
        width = sum(self.assembly_size(a)[0] for a in self.core[0])
        height = sum(self.assembly_size(a)[1] for a in self.core[:, 0])
        return width, height

    def validate(self):
        """Check that the plan is a valid MPACT partition. Raises ValueError."""
        if self.core is None:
            raise ValueError('The core has not been set')
        for r, ids in enumerate(self.rtms):
            for row in ids:
                hs = {self.coarse_cells[i].height for i in row}
                if any(not _close(h, next(iter(hs))) for h in hs):
                    raise ValueError(f'RTM {r}: coarse cells in a row must have the '
                                     'same height')
            for col in ids.T:
                ws = {self.coarse_cells[i].width for i in col}
                if any(not _close(w, next(iter(ws))) for w in ws):
                    raise ValueError(f'RTM {r}: coarse cells in a column must have '
                                     'the same width')
        for lat, ids in enumerate(self.lattices):
            w0, h0 = self.rtm_size(ids.flat[0])
            for i in ids.flat:
                w, h = self.rtm_size(i)
                if not (_close(w, w0) and _close(h, h0)):
                    raise ValueError(f'Lattice {lat}: all RTMs must have the same '
                                     f'size, but RTM {i} is {w} x {h} and RTM '
                                     f'{ids.flat[0]} is {w0} x {h0}')
        for row in self.core:
            hs = [self.assembly_size(a)[1] for a in row]
            if any(not _close(h, hs[0]) for h in hs):
                raise ValueError('Assemblies in a row of the core must have the '
                                 'same height')
        for col in self.core.T:
            ws = [self.assembly_size(a)[0] for a in col]
            if any(not _close(w, ws[0]) for w in ws):
                raise ValueError('Assemblies in a column of the core must have the '
                                 'same width')
        sizes = {tuple(np.round(self.rtm_size(i), 9)) for i in range(len(self.rtms))
                 if self._rtm_used(i)}
        if len(sizes) > 1:
            warnings.warn(f'The RTMs of the core have different sizes {sorted(sizes)}. '
                          'MPACT modular ray tracing requires RTMs of equal size.')

    def _rtm_used(self, rtm_id):
        return any(rtm_id in self.lattices[self.assemblies[a][0]]
                   for a in set(self.core.flat))

    def coarse_cell_boxes(self):
        """Return the box ``(xmin, ymin, xmax, ymax)``, in the coordinates of
        the OpenMC geometry, of the first occurrence of each coarse cell.

        UM2 meshes each unique coarse cell once, at its first occurrence in the
        order in which UM2 traverses the core (assemblies from the bottom-left,
        row by row, and likewise for lattices and RTMs). Returns a dict mapping
        coarse cell ID to box.
        """
        if self.core is None:
            raise ValueError('The core has not been set')
        boxes = {}
        seen_asy, seen_lat, seen_rtm = set(), set(), set()
        ox, oy = self.origin
        core = self.core[::-1]      # row 0 at the bottom
        y_asy = 0.0
        for iy, row in enumerate(core):
            x_asy = 0.0
            for ix, asy in enumerate(row):
                w_asy, h_asy = self.assembly_size(asy)
                if asy not in seen_asy:
                    seen_asy.add(asy)
                    lat = self.assemblies[asy][0]
                    if lat not in seen_lat:
                        seen_lat.add(lat)
                        self._lattice_boxes(lat, x_asy, y_asy, boxes, seen_rtm)
                x_asy += w_asy
            y_asy += self.assembly_size(row[0])[1]
        return {cc: (ox + b[0], oy + b[1], ox + b[2], oy + b[3])
                for cc, b in boxes.items()}

    def _lattice_boxes(self, lat, x0, y0, boxes, seen_rtm):
        ids = self.lattices[lat][::-1]
        w_rtm, h_rtm = self.rtm_size(ids[0, 0])
        for iy, row in enumerate(ids):
            for ix, rtm in enumerate(row):
                if rtm in seen_rtm:
                    continue
                seen_rtm.add(rtm)
                cc_ids = self.rtms[rtm][::-1]
                y = y0 + iy * h_rtm
                for cy, cc_row in enumerate(cc_ids):
                    x = x0 + ix * w_rtm
                    for cx, cc in enumerate(cc_row):
                        c = self.coarse_cells[cc]
                        if cc not in boxes:
                            boxes[cc] = (x, y, x + c.width, y + c.height)
                        x += c.width
                    y += self.coarse_cells[cc_row[0]].height

    # --------------------------------------------------------------------------
    # Conversion
    # --------------------------------------------------------------------------

    def to_um2(self):
        """Create the (C++) UM2 MPACT model for this plan (without materials or
        meshes)."""
        from . import _core
        self.validate()
        model = _core().mpact.Model()
        for cc in self.coarse_cells:
            model.add_coarse_cell((cc.width, cc.height))
        for ids in self.rtms:
            model.add_rtm(ids.tolist())
        for ids in self.lattices:
            model.add_lattice(ids.tolist())
        # 2D models use UM2's special case z = (-0.5, 0.5), which places the
        # 2D cut plane (the plane of the geometry) at z = 0. The axial extent
        # of the OpenMC model (self.z) is informational only.
        for ids in self.assemblies:
            model.add_assembly(ids, [-0.5, 0.5])
        model.add_core(self.core.tolist())
        return model

    def __str__(self):
        lines = ['MPACT partition plan']
        if self.core is None:
            lines.append('  (no core)')
            return '\n'.join(lines)
        w, h = self.size
        lines.append(f'  core: {self.core.shape[1]} x {self.core.shape[0]} assemblies, '
                     f'{w:.6g} x {h:.6g} cm, lower-left corner at '
                     f'({self.origin[0]:.6g}, {self.origin[1]:.6g}), '
                     f'z in [{self.z[0]:.6g}, {self.z[1]:.6g}]')
        for row in self.core:
            lines.append('    ' + ' '.join(f'{a:3d}' for a in row))
        for a, lats in enumerate(self.assemblies):
            lat = lats[0]
            ids = self.lattices[lat]
            rw, rh = self.rtm_size(ids[0, 0])
            lines.append(f'  assembly {a}: lattice {lat} of {ids.shape[1]} x '
                         f'{ids.shape[0]} RTMs ({rw:.6g} x {rh:.6g} cm), '
                         f'{len(np.unique(ids))} unique')
        n_rtm_cc = {tuple(ids.shape) for ids in self.rtms}
        lines.append(f'  {len(self.rtms)} RTMs, of '
                     + ', '.join(f'{nx} x {ny}' for ny, nx in sorted(n_rtm_cc))
                     + ' coarse cells')
        lines.append(f'  {len(self.coarse_cells)} unique coarse cells:')
        for i, cc in enumerate(self.coarse_cells):
            lines.append(f'    {i:4d}: {cc.width:.6g} x {cc.height:.6g} cm  {cc.label}')
        for note in self.notes:
            lines.append(f'  note: {note}')
        return '\n'.join(lines)


# ==============================================================================
# Inference
# ==============================================================================

def _root_box(geometry):
    from .build import _root_box as root_box
    b = root_box(geometry)
    return (b.xmin, b.ymin, b.xmax, b.ymax)


def _rect_lattice_extent(lattice: RectLattice):
    nx, ny = lattice.shape
    x0, y0 = lattice.lower_left[0], lattice.lower_left[1]
    return (x0, y0, x0 + nx * lattice.pitch[0], y0 + ny * lattice.pitch[1])


def _boxes_close(a, b):
    scale = max(abs(v) for v in a + b) or 1.0
    return all(_close(x, y, scale) for x, y in zip(a, b))


def _structural_lattice(universe, box):
    """If a universe is exactly filled by a RectLattice that covers the box
    (in the universe's coordinates), return the lattice."""
    if not isinstance(universe, Universe) or len(universe.cells) != 1:
        return None
    cell = next(iter(universe.cells.values()))
    if cell.fill_type != 'lattice' or not isinstance(cell.fill, RectLattice):
        return None
    if cell.translation is not None and np.any(cell.translation[:2] != 0):
        return None
    if cell.rotation is not None:
        return None
    lattice = cell.fill
    if not _boxes_close(_rect_lattice_extent(lattice), tuple(box)):
        return None
    if cell.region is not None:
        bb = cell.region.bounding_box
        rbox = (bb.lower_left[0], bb.lower_left[1], bb.upper_right[0], bb.upper_right[1])
        # The region must not cut the lattice
        if not (rbox[0] <= box[0] + _RTOL and rbox[1] <= box[1] + _RTOL and
                rbox[2] >= box[2] - _RTOL and rbox[3] >= box[3] - _RTOL):
            return None
        if not _region_is_box(cell.region):
            return None
    return lattice


def _region_is_box(region):
    """Whether a region is an intersection of half-spaces of x/y-planes (and
    axial planes), i.e. an axis-aligned box."""
    from .openmc.region import Intersection
    nodes = list(region) if isinstance(region, Intersection) else [region]
    for node in nodes:
        if isinstance(node, Intersection):
            if not _region_is_box(node):
                return False
        elif not (isinstance(node, Halfspace) and
                  (isinstance(node.surface, (XPlane, YPlane)) or
                   getattr(node.surface, 'is_axial', False))):
            return False
    return True


def _is_homogeneous(universe):
    """Return the material if every cell of a universe (recursively) is filled
    with the same material, otherwise None."""
    materials = set()
    for cell in universe.get_all_cells().values():
        if cell.fill_type == 'material':
            materials.add(cell.fill)
        elif cell.fill_type == 'void':
            return None
    if len(materials) == 1 and all(c.fill_type in ('material', 'universe', 'lattice')
                                   for c in universe.get_all_cells().values()):
        return next(iter(materials))
    return None


def _name(obj):
    return f'{type(obj).__name__} {obj.id}' + (f' "{obj.name}"' if obj.name else '')


class _Inference:
    def __init__(self, plan, rtm, rtm_size):
        self.plan = plan
        self.rtm_mode = rtm
        self.rtm_size = rtm_size
        self.cc_by_key = {}
        self.rtm_by_key = {}
        self.lat_by_key = {}
        self.asy_by_key = {}

    def coarse_cell(self, width, height, key, label):
        if key not in self.cc_by_key:
            self.cc_by_key[key] = self.plan.add_coarse_cell(width, height, label, key)
        return self.cc_by_key[key]

    def rtm(self, cc_ids):
        key = tuple(map(tuple, cc_ids))
        if key not in self.rtm_by_key:
            self.rtm_by_key[key] = self.plan.add_rtm(cc_ids)
        return self.rtm_by_key[key]

    def lattice(self, rtm_ids):
        key = tuple(map(tuple, rtm_ids))
        if key not in self.lat_by_key:
            self.lat_by_key[key] = self.plan.add_lattice(rtm_ids)
        return self.lat_by_key[key]

    def assembly(self, lattice_id):
        if lattice_id not in self.asy_by_key:
            self.asy_by_key[lattice_id] = self.plan.add_assembly([lattice_id])
        return self.asy_by_key[lattice_id]

    def element_cells(self, universe, width, height):
        """Coarse cells (rows top to bottom) for a lattice element of the given
        size containing a universe, subdivided to the RTM size."""
        rx, ry = self.rtm_size
        nx = max(1, round(width / rx))
        ny = max(1, round(height / ry))
        if not (_close(nx * rx, width) and _close(ny * ry, height)):
            raise ValueError(
                f'The lattice element containing {_name(universe)} is {width:.6g} x '
                f'{height:.6g} cm, which is not a multiple of the RTM size '
                f'{rx:.6g} x {ry:.6g} cm. Pass rtm_size to infer_hierarchy, or build '
                'the Plan by hand.')
        if nx == 1 and ny == 1:
            key = ('element', universe.id, round(width, 9), round(height, 9))
            return [[self.coarse_cell(width, height, key, _name(universe))]]
        material = _is_homogeneous(universe)
        cw, ch = width / nx, height / ny
        rows = []
        for j in range(ny):
            row = []
            for i in range(nx):
                if material is not None:
                    key = ('homogeneous', material.id, round(cw, 9), round(ch, 9))
                    label = f'homogeneous {_name(material)}'
                else:
                    key = ('part', universe.id, round(width, 9), round(height, 9), i, j)
                    label = f'{_name(universe)}, part ({i}, {ny - 1 - j})'
                row.append(self.coarse_cell(cw, ch, key, label))
            rows.append(row)
        return rows

    def assembly_from_universe(self, universe, width, height):
        """Assembly ID for a core element (a universe of the given size)."""
        box = (-width / 2, -height / 2, width / 2, height / 2)
        lattice = _structural_lattice(universe, box)
        if lattice is None:
            # A single element, subdivided into RTMs
            cells = self.element_cells(universe, width, height)
        else:
            cells = self.lattice_cells(lattice)
        return self.assembly(self.lattice(self.rtms_from_cells(cells)))

    def lattice_cells(self, lattice: RectLattice):
        """Coarse cell IDs (rows top to bottom) covering a pin lattice."""
        px, py = lattice.pitch[0], lattice.pitch[1]
        blocks = [[self.element_cells(u, px, py) for u in row]
                  for row in lattice.universes]
        cells = []
        for block_row in blocks:
            for k in range(len(block_row[0])):
                cells.append([c for block in block_row for c in block[k]])
        return cells

    def rtms_from_cells(self, cells):
        """Group a grid of coarse cells (rows top to bottom) into RTMs."""
        cells = np.array(cells, dtype=int)
        if self.rtm_mode == 'assembly':
            return [[self.rtm(cells)]]
        # One RTM per coarse cell
        return [[self.rtm([[c]]) for c in row] for row in cells]


def _grid_lines(geometry, box, target):
    """x and y lines of a rectilinear coarse grid for a general geometry."""
    xs = {box[0], box[2]}
    ys = {box[1], box[3]}
    for cell in geometry.root_universe.cells.values():
        if cell.region is not None:
            for surf in cell.region.get_surfaces().values():
                if isinstance(surf, XPlane) and box[0] < surf.x0 < box[2]:
                    xs.add(surf.x0)
                elif isinstance(surf, YPlane) and box[1] < surf.y0 < box[3]:
                    ys.add(surf.y0)
        if cell.fill_type == 'lattice' and isinstance(cell.fill, RectLattice):
            lat = cell.fill
            dx = dy = 0.0
            if cell.translation is not None:
                dx, dy = cell.translation[0], cell.translation[1]
            x0, y0, x1, y1 = _rect_lattice_extent(lat)
            nx, ny = lat.shape
            for i in range(nx + 1):
                x = dx + x0 + i * lat.pitch[0]
                if box[0] < x < box[2]:
                    xs.add(x)
            for j in range(ny + 1):
                y = dy + y0 + j * lat.pitch[1]
                if box[1] < y < box[3]:
                    ys.add(y)

    def refine(values, lo, hi):
        values = sorted(values)
        merged = [values[0]]
        for v in values[1:]:
            if v - merged[-1] > _RTOL * (hi - lo):
                merged.append(v)
        merged[-1] = hi
        lines = [merged[0]]
        for a, b in zip(merged[:-1], merged[1:]):
            n = max(1, round((b - a) / target)) if target else 1
            lines.extend(a + (b - a) * k / n for k in range(1, n + 1))
        return lines

    return refine(xs, box[0], box[2]), refine(ys, box[1], box[3])


def _smallest_pitch(geometry):
    pitches = []
    for lat in geometry.get_all_lattices().values():
        if isinstance(lat, RectLattice):
            pitches.append(min(lat.pitch[0], lat.pitch[1]))
        elif isinstance(lat, HexLattice):
            pitches.append(lat.pitch[0])
    return min(pitches) if pitches else None


def grid_plan(geometry, coarse_cell_size=None, *, x=None, y=None) -> Plan:
    """Create a plan with a single RTM that is a rectilinear grid of coarse
    cells covering the geometry.

    Parameters
    ----------
    geometry : um2.openmc.Geometry
    coarse_cell_size : float, optional
        Target coarse cell size [cm]. The grid lines include the x/y-planes that
        bound the root cells and the element boundaries of rectangular lattices
        that fill root cells, and the spaces between them are subdivided so that
        coarse cells are no larger than about this size. Defaults to the
        smallest lattice pitch in the geometry, or a single coarse cell if there
        are no lattices.
    x, y : sequence of float, optional
        Explicit grid lines (in the coordinates of the geometry), overriding
        the automatic lines.
    """
    box = _root_box(geometry)
    if coarse_cell_size is None:
        coarse_cell_size = _smallest_pitch(geometry)
    auto_x, auto_y = _grid_lines(geometry, box, coarse_cell_size)
    xs = sorted(x) if x is not None else auto_x
    ys = sorted(y) if y is not None else auto_y
    if not (_close(xs[0], box[0]) and _close(xs[-1], box[2]) and
            _close(ys[0], box[1]) and _close(ys[-1], box[3])):
        raise ValueError(f'The grid lines must span the geometry, {box}')
    plan = Plan(origin=(box[0], box[1]))
    rows = []
    for j in reversed(range(len(ys) - 1)):
        row = []
        for i in range(len(xs) - 1):
            row.append(plan.add_coarse_cell(xs[i + 1] - xs[i], ys[j + 1] - ys[j],
                                            f'grid ({i}, {j})', ('grid', i, j)))
        rows.append(row)
    rtm = plan.add_rtm(rows)
    lat = plan.add_lattice([[rtm]])
    asy = plan.add_assembly([lat])
    plan.set_core([[asy]])
    plan.notes.append(f'rectilinear coarse grid of {len(xs) - 1} x {len(ys) - 1} '
                      'cells in a single RTM')
    return plan


def infer_hierarchy(geometry, *, rtm='pin', rtm_size=None, core_lattice=None,
                    coarse_cell_size=None) -> Plan:
    """Infer the MPACT partition of an OpenMC geometry.

    Rectangular lattices that exactly fill the root universe define the
    partition:

    * If the root universe is filled by a lattice of lattices (e.g. a core of
      assemblies of pins), the outer lattice is the core and each element is
      an assembly.
    * If the root universe is filled by a lattice of pins, the core is a
      single assembly.
    * Each pin (lattice element) is a coarse cell. Elements that are larger
      than the RTM size (e.g. a reflector assembly defined as a single
      element) are subdivided into coarse cells of the RTM size.
    * Coarse cells are shared only when they contain the same universe in an
      element of the same size, so each unique pin is meshed once.

    Any other geometry (e.g. hexagonal lattices, or a lattice surrounded by
    other cells) is partitioned by a rectilinear grid, see :func:`grid_plan`.

    Parameters
    ----------
    geometry : um2.openmc.Geometry or um2.openmc.Model
        The geometry to partition
    rtm : {'pin', 'assembly'}
        Whether each RTM is a single pin (coarse cell), or a whole assembly
        lattice
    rtm_size : tuple of float, optional
        (width, height) of the RTMs when ``rtm='pin'``. Defaults to the
        smallest pin pitch.
    core_lattice : um2.openmc.RectLattice, optional
        Lattice to use as the core. By default, this is inferred.
    coarse_cell_size : float, optional
        Coarse cell size of the grid used for geometries without a lattice
        structure (see :func:`grid_plan`)

    Returns
    -------
    Plan
    """
    from .build import _check_axial
    from .openmc.model import Model
    if isinstance(geometry, Model):
        geometry = geometry.geometry
    if not isinstance(geometry, Geometry):
        raise TypeError('infer_hierarchy expects a um2.openmc.Geometry or Model')
    if rtm not in ('pin', 'assembly'):
        raise ValueError("rtm must be 'pin' or 'assembly'")

    axial = _check_axial(geometry)
    z = (axial.zmin, axial.zmax) if axial.is_bounded else (-0.5, 0.5)
    box = _root_box(geometry)
    root = geometry.root_universe
    top = core_lattice if core_lattice is not None else _structural_lattice(root, box)
    if top is None:
        plan = grid_plan(geometry, coarse_cell_size)
        plan.z = z
        plan.notes.insert(0, 'the root universe is not exactly filled by a '
                          'rectangular lattice')
        return plan
    if not _boxes_close(_rect_lattice_extent(top), box):
        raise ValueError(f'The core lattice {_name(top)} does not cover the geometry')

    # Is the top lattice a core (a lattice of lattices) or a single assembly?
    px, py = top.pitch[0], top.pitch[1]
    element_box = (-px / 2, -py / 2, px / 2, py / 2)
    sub_lattices = {u.id: _structural_lattice(u, element_box)
                    for u in top.get_unique_universes().values()}
    is_core = any(lat is not None for lat in sub_lattices.values())

    plan = Plan(origin=(box[0], box[1]), z=z)
    if rtm_size is None:
        pitches = [(lat.pitch[0], lat.pitch[1]) for lat in sub_lattices.values()
                   if lat is not None] if is_core else [(px, py)]
        rtm_size = (min(p[0] for p in pitches), min(p[1] for p in pitches))
    inference = _Inference(plan, rtm, tuple(rtm_size))

    if is_core:
        core = [[inference.assembly_from_universe(u, px, py) for u in row]
                for row in top.universes]
        plan.notes.append(f'core lattice: {_name(top)}')
    else:
        cells = inference.lattice_cells(top)
        core = [[inference.assembly(inference.lattice(inference.rtms_from_cells(cells)))]]
        plan.notes.append(f'single assembly: {_name(top)}')
    plan.set_core(core)
    plan.validate()
    return plan
