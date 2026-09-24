"""Translation of OpenMC CSG geometry into gmsh faces."""

import math

import numpy as np
import pytest

import um2
import um2.openmc as openmc
from um2.build import material_names
from um2.errors import UM2NotSupportedError


def sampled_areas(geometry, box, n=40000, seed=1):
    """Estimate material areas by locating random points with Geometry.find,
    which is independent of the gmsh builder."""
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = box
    pts = np.column_stack([rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n),
                           np.zeros(n)])
    names = material_names(geometry.get_all_materials().values())
    counts = {}
    for p in pts:
        path = geometry.find(p)
        if not path:
            continue
        cell = path[-1]
        name = names[cell.fill] if cell.fill_type == 'material' else 'Void'
        counts[name] = counts.get(name, 0) + 1
    area = (xmax - xmin) * (ymax - ymin)
    return {k: v / n * area for k, v in counts.items()}, area, n


def check_against_sampling(geometry, result):
    areas = result.material_areas()
    est, total, n = sampled_areas(geometry, result.bounding_box)
    for name in set(areas) | set(est):
        a, e = areas.get(name, 0.0), est.get(name, 0.0)
        p = e / total
        sigma = total * math.sqrt(max(p * (1 - p), 1 / n) / n)
        assert abs(a - e) < 5 * sigma + 1e-9, (name, a, e, sigma)


def pin_universe(fuel, water, r=0.4, gap=None):
    cells = []
    fo = openmc.ZCylinder(r=r)
    cells.append(openmc.Cell(fill=fuel, region=-fo))
    if gap is not None:
        g = openmc.ZCylinder(r=gap)
        cells.append(openmc.Cell(region=+fo & -g))
        fo = g
    cells.append(openmc.Cell(fill=water, region=+fo))
    return openmc.Universe(cells=cells)


@pytest.fixture
def mats():
    return {n: openmc.Material(name=n) for n in ('fuel', 'clad', 'water', 'steel')}


def test_pin_cell_areas(gmsh_session, mats):
    fo, ci, co = (openmc.ZCylinder(r=r) for r in (0.39, 0.40, 0.46))
    box = openmc.model.RectangularPrism(1.26, 1.26, boundary_type='reflective')
    u = openmc.Universe(cells=[
        openmc.Cell(fill=mats['fuel'], region=-fo),
        openmc.Cell(region=+fo & -ci),
        openmc.Cell(fill=mats['clad'], region=+ci & -co),
        openmc.Cell(fill=mats['water'], region=+co & -box)])
    r = um2.build_geometry(openmc.Geometry(u))
    a = r.material_areas()
    assert a['fuel'] == pytest.approx(math.pi * 0.39**2, rel=1e-9)
    assert a['Void'] == pytest.approx(math.pi * (0.40**2 - 0.39**2), rel=1e-9)
    assert a['clad'] == pytest.approx(math.pi * (0.46**2 - 0.40**2), rel=1e-9)
    assert a['water'] == pytest.approx(1.26**2 - math.pi * 0.46**2, rel=1e-9)
    assert r.boundary_types == {'reflective'}
    assert r.bounding_box == pytest.approx((-0.63, -0.63, 0.63, 0.63))
    gmsh = gmsh_session
    names = sorted(gmsh.model.getPhysicalName(*g) for g in gmsh.model.getPhysicalGroups())
    assert names == ['Material_Void', 'Material_clad', 'Material_fuel', 'Material_water']


def test_conformal(gmsh_session, mats):
    """Neighbouring faces share their boundary curves."""
    u = pin_universe(mats['fuel'], mats['water'])
    box = openmc.model.RectangularPrism(1.26, 1.26)
    root = openmc.Universe(cells=[openmc.Cell(fill=u, region=-box)])
    um2.build_geometry(openmc.Geometry(root))
    gmsh = gmsh_session
    faces = gmsh.model.getEntities(2)
    assert len(faces) == 2
    b1 = {abs(t) for _, t in gmsh.model.getBoundary([faces[0]], oriented=False)}
    b2 = {abs(t) for _, t in gmsh.model.getBoundary([faces[1]], oriented=False)}
    assert b1 & b2   # the circle is shared
    # Unused curves and points are removed
    assert len(gmsh.model.getEntities(1)) == len(b1 | b2)


def test_rect_lattice(gmsh_session, mats):
    pin = pin_universe(mats['fuel'], mats['water'])
    gt = pin_universe(mats['water'], mats['water'], r=0.5)
    lat = openmc.RectLattice()
    lat.lower_left = (-1.89, -1.89)
    lat.pitch = (1.26, 1.26)
    lat.universes = [[pin, pin, pin], [pin, gt, pin], [pin, pin, pin]]
    box = openmc.model.RectangularPrism(3.78, 3.78, boundary_type='reflective')
    geom = openmc.Geometry([openmc.Cell(fill=lat, region=-box)])
    r = um2.build_geometry(geom)
    a = r.material_areas()
    assert a['fuel'] == pytest.approx(8 * math.pi * 0.16, rel=1e-9)
    assert a['water'] == pytest.approx(3.78**2 - 8 * math.pi * 0.16, rel=1e-9)
    check_against_sampling(geom, r)


def test_lattice_partially_covered_by_cell(gmsh_session, mats):
    """A lattice clipped by the region of the cell it fills (partial elements)."""
    pin = pin_universe(mats['fuel'], mats['water'], r=0.5)
    lat = openmc.RectLattice()
    lat.lower_left = (-2.0, -2.0)
    lat.pitch = (1.0, 1.0)
    lat.universes = [[pin] * 4] * 4
    region = -openmc.ZCylinder(r=1.8, boundary_type='vacuum')
    geom = openmc.Geometry([openmc.Cell(fill=lat, region=region)])
    r = um2.build_geometry(geom)
    assert sum(r.material_areas().values()) == pytest.approx(math.pi * 1.8**2, rel=1e-6)
    check_against_sampling(geom, r)


@pytest.mark.parametrize('orientation', ['y', 'x'])
def test_hex_lattice_with_outer(gmsh_session, mats, orientation):
    small = pin_universe(mats['fuel'], mats['water'], r=0.25)
    big = pin_universe(mats['steel'], mats['water'], r=0.5)
    outer = openmc.Universe(cells=[openmc.Cell(fill=mats['water'])])
    lat = openmc.HexLattice()
    lat.center = (0., 0.)
    lat.pitch = (1.25,)
    lat.orientation = orientation
    lat.outer = outer
    lat.universes = [[big] + [small] * 17, [big] + [small] * 11, [big] + [small] * 5, [big]]
    edge = 4 * 1.25
    prism = openmc.model.HexagonalPrism(edge_length=edge, orientation=orientation,
                                        boundary_type='vacuum')
    geom = openmc.Geometry([openmc.Cell(fill=lat, region=-prism)])
    r = um2.build_geometry(geom)
    a = r.material_areas()
    assert a['fuel'] == pytest.approx(33 * math.pi * 0.25**2, rel=1e-9)
    assert a['steel'] == pytest.approx(4 * math.pi * 0.5**2, rel=1e-9)
    assert sum(a.values()) == pytest.approx(1.5 * math.sqrt(3) * edge**2, rel=1e-9)
    check_against_sampling(geom, r)


def test_union_complement_and_translation(gmsh_session, mats):
    """BWR-like control blade (union), water fill (complements) and a pin that
    is placed by translating a universe (CANDU-like)."""
    x0, x1 = openmc.XPlane(0.0, boundary_type='reflective'), \
        openmc.XPlane(10.0, boundary_type='reflective')
    y0, y1 = openmc.YPlane(0.0, boundary_type='reflective'), \
        openmc.YPlane(10.0, boundary_type='reflective')
    everything = +x0 & -x1 & +y0 & -y1
    blade_h = +y0 & -openmc.YPlane(1.0) & +x0 & -x1
    blade_v = +x0 & -openmc.XPlane(1.0) & +y0 & -y1
    blade = blade_h | blade_v
    pin = pin_universe(mats['fuel'], mats['clad'], r=0.5)
    boundary = openmc.ZCylinder(x0=5.0, y0=5.0, r=1.0)
    pin_cell = openmc.Cell(fill=pin, region=-boundary)
    pin_cell.translation = (5.0, 5.0, 0.0)
    water = openmc.Cell(fill=mats['water'], region=everything & ~blade)
    water.region &= +boundary
    geom = openmc.Geometry([openmc.Cell(fill=mats['steel'], region=blade), water,
                            pin_cell])
    r = um2.build_geometry(geom)
    a = r.material_areas()
    assert a['steel'] == pytest.approx(19.0, rel=1e-9)
    assert a['fuel'] == pytest.approx(math.pi * 0.25, rel=1e-9)
    assert a['clad'] == pytest.approx(math.pi * 0.75, rel=1e-9)
    assert a['water'] == pytest.approx(100 - 19 - math.pi, rel=1e-9)
    check_against_sampling(geom, r)


def test_rotated_universe(gmsh_session, mats):
    """A square rotated by 45 degrees about z."""
    square = openmc.model.RectangularPrism(1.0, 1.0)
    inner = openmc.Universe(cells=[openmc.Cell(fill=mats['fuel'], region=-square),
                                   openmc.Cell(fill=mats['water'], region=+square)])
    cell = openmc.Cell(fill=inner, region=-openmc.model.RectangularPrism(4.0, 4.0))
    cell.rotation = (0.0, 0.0, 45.0)
    geom = openmc.Geometry([cell])
    r = um2.build_geometry(geom)
    a = r.material_areas()
    assert a['fuel'] == pytest.approx(1.0, rel=1e-9)
    bb = gmsh_session.model.getBoundingBox(2, r.material_surfaces['fuel'][0])
    assert bb[3] - bb[0] == pytest.approx(math.sqrt(2), abs=1e-6)
    check_against_sampling(geom, r)


def test_general_planes(gmsh_session, mats):
    sector = openmc.model.CylinderSector(1.0, 2.0, 30.0, 150.0)
    octagon = openmc.model.IsogonalOctagon((0.0, 0.0), 3.0, 3.5)
    geom = openmc.Geometry([openmc.Cell(fill=mats['fuel'], region=-sector),
                            openmc.Cell(fill=mats['water'], region=-octagon & +sector)])
    r = um2.build_geometry(geom)
    a = r.material_areas()
    assert a['fuel'] == pytest.approx(math.pi * 3 * 120 / 360, rel=1e-9)
    check_against_sampling(geom, r)


def test_axial_bounds(gmsh_session, mats):
    box = openmc.model.RectangularPrism(1.0, 1.0, boundary_type='reflective')
    z0 = openmc.ZPlane(-10.0, boundary_type='reflective')
    z1 = openmc.ZPlane(10.0, boundary_type='reflective')
    geom = openmc.Geometry([openmc.Cell(fill=mats['water'], region=-box & +z0 & -z1)])
    r = um2.build_geometry(geom)
    assert (r.axial_bounds.zmin, r.axial_bounds.zmax) == (-10.0, 10.0)
    # Vacuum axial boundaries are ignored with a warning
    gmsh_session.model.add('vacuum')
    z1.boundary_type = 'vacuum'
    with pytest.warns(UserWarning, match='vacuum boundary in z'):
        um2.build_geometry(geom)


def test_axial_layers_raise(gmsh_session, mats):
    box = openmc.model.RectangularPrism(1.0, 1.0)
    z = openmc.ZPlane(0.0)
    geom = openmc.Geometry([openmc.Cell(fill=mats['fuel'], region=-box & -z),
                            openmc.Cell(fill=mats['water'], region=-box & +z)])
    with pytest.raises(UM2NotSupportedError):
        um2.build_geometry(geom)
    geom2 = openmc.Geometry([openmc.Cell(fill=mats['fuel'], region=-box & (-z | +z))])
    with pytest.raises(UM2NotSupportedError):
        um2.build_geometry(geom2)


def test_unbounded_root_raises(gmsh_session, mats):
    geom = openmc.Geometry([openmc.Cell(fill=mats['water'], region=+openmc.XPlane(0.0))])
    with pytest.raises(ValueError, match='bounded'):
        um2.build_geometry(geom)


def test_material_names():
    a = openmc.Material(name='1.6% Fuel')
    b = openmc.Material(name='1.6% Fuel')
    c = openmc.Material()
    names = material_names([a, b, c])
    assert names[a] == '1.6_Fuel'
    assert names[b] == f'1.6_Fuel_{b.id}'
    assert names[c] == f'Material{c.id}'
