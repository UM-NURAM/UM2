"""Surfaces, regions, cells, universes and lattices (no gmsh required)."""

import math

import numpy as np
import pytest

import um2.openmc as openmc
from um2.errors import UM2NotSupportedError


def test_halfspaces_and_containment():
    cyl = openmc.ZCylinder(x0=1.0, y0=2.0, r=0.5)
    assert (1.0, 2.0, 0.0) in -cyl
    assert (1.6, 2.0, 0.0) in +cyl
    x = openmc.XPlane(0.0)
    y = openmc.YPlane(y0=1.0)
    region = +x & -y & +cyl
    assert (3.0, 0.0, 0.0) in region
    assert (-1.0, 0.0, 0.0) not in region
    assert (0.5, 0.5, 0.0) in (-x | -y)
    assert (0.5, 0.5, 0.0) not in ~(-x | -y) | -x
    assert str(+x & -y) == f'({x.id} -{y.id})'
    r = openmc.Region.from_expression(f'{x.id} -{y.id} | ~(-{cyl.id})',
                                      {x.id: x, y.id: y, cyl.id: cyl})
    assert (5.0, 5.0, 0.0) in r


def test_region_iand():
    cyl = openmc.ZCylinder(r=1.0)
    region = -openmc.model.RectangularPrism(4.0, 4.0)
    region &= +cyl
    assert (0, 0, 0) not in region and (1.5, 0, 0) in region


def test_bounding_boxes():
    box = openmc.model.RectangularPrism(2.0, 4.0, origin=(1.0, 0.0),
                                        boundary_type='reflective')
    bb = (-box).bounding_box
    assert np.allclose(bb.lower_left[:2], (0.0, -2.0))
    assert np.allclose(bb.upper_right[:2], (2.0, 2.0))
    assert box.boundary_type == 'reflective'
    cyl = openmc.ZCylinder(r=3.0)
    assert np.allclose((-cyl).bounding_box.upper_right[:2], (3.0, 3.0))
    assert np.all(np.isinf((+cyl).bounding_box.upper_right))


def test_surface_coefficients_and_ids():
    s = openmc.ZCylinder(r=0.39, name='fuel')
    assert s.coefficients == {'x0': 0.0, 'y0': 0.0, 'r': 0.39}
    assert s.type == 'z-cylinder'
    t = openmc.XPlane(1.0)
    assert t.id == s.id + 1
    with pytest.raises(ValueError):
        openmc.ZCylinder(r=-1.0)
    with pytest.raises(ValueError):
        s.boundary_type = 'bogus'
    p = openmc.Plane.from_points((0, 0, 0), (1, 1, 0), (0, 0, 1))
    assert p.c == pytest.approx(0.0)


@pytest.mark.parametrize('cls, kwargs', [
    ('Sphere', {'r': 1.0}), ('Cone', {}), ('XCylinder', {'r': 1.0}),
    ('YCylinder', {'r': 1.0}), ('Quadric', {}), ('ZTorus', {}),
])
def test_3d_surfaces_raise(cls, kwargs):
    with pytest.raises(UM2NotSupportedError):
        getattr(openmc, cls)(**kwargs)


def test_oblique_plane_raises():
    with pytest.raises(UM2NotSupportedError):
        openmc.Plane(a=1.0, b=0.0, c=1.0, d=0.0)
    # Axial planes and planes containing the z-direction are fine
    openmc.Plane(a=0.0, b=0.0, c=1.0, d=1.0)
    openmc.Plane(a=1.0, b=2.0, c=0.0, d=1.0)


def test_rotation_about_z_only():
    cell = openmc.Cell()
    cell.rotation = (0.0, 0.0, 30.0)
    with pytest.raises(UM2NotSupportedError):
        cell.rotation = (10.0, 0.0, 0.0)
    with pytest.raises(UM2NotSupportedError):
        openmc.model.RectangularPrism(1.0, 1.0, axis='x')


def test_distributed_materials_raise():
    m = openmc.Material()
    with pytest.raises(UM2NotSupportedError):
        openmc.Cell(fill=[m, m])


def test_composite_surfaces():
    hexagon = openmc.model.HexagonalPrism(edge_length=1.0, orientation='y')
    assert (0.0, 0.99, 0.0) in -hexagon
    assert (0.87, 0.0, 0.0) not in -hexagon
    hexagon_x = openmc.model.HexagonalPrism(edge_length=1.0, orientation='x')
    assert (0.99, 0.0, 0.0) in -hexagon_x
    assert (0.0, 0.87, 0.0) not in -hexagon_x
    sector = openmc.model.CylinderSector(1.0, 2.0, 0.0, 90.0)
    assert (1.0, 1.0, 0.0) in -sector
    assert (-1.0, 1.0, 0.0) not in -sector
    octagon = openmc.model.IsogonalOctagon((0.0, 0.0), 1.0, 1.2)
    assert (0.0, 0.0, 0.0) in -octagon
    assert (0.95, 0.95, 0.0) not in -octagon
    with pytest.warns(FutureWarning):
        region = openmc.model.rectangular_prism(1.0, 1.0)
    assert (0.0, 0.0, 0.0) in region


def test_pin_helper():
    fuel = openmc.Material(name='fuel')
    fuel.volume = 10.0
    water = openmc.Material(name='water')
    surfs = [openmc.ZCylinder(r=r) for r in (0.4, 0.45)]
    univ = openmc.model.pin(surfs, [fuel, water, water], subdivisions={0: 4})
    cells = list(univ.cells.values())
    assert len(cells) == 6
    assert fuel.volume == pytest.approx(2.5)
    # Equal-area rings
    radii = sorted(s.r for c in cells if c.region is not None
                   for s in c.region.get_surfaces().values())
    assert radii[0] == pytest.approx(0.2)


def test_rect_lattice_indexing():
    u = [openmc.Universe(name=str(i)) for i in range(6)]
    lat = openmc.RectLattice()
    lat.lower_left = (-1.5, -1.0)
    lat.pitch = (1.0, 1.0)
    lat.universes = [[u[0], u[1], u[2]], [u[3], u[4], u[5]]]   # row 0 is the top
    assert lat.shape == (3, 2)
    assert lat.get_universe((0, 1)) is u[0]
    assert lat.get_universe((2, 0)) is u[5]
    idx, local = lat.find_element((1.2, -0.7, 0.0))
    assert idx == (2, 0)
    assert np.allclose(local[:2], (0.2, -0.2))
    # numpy object arrays and aliased rows work too
    arr = np.empty((17, 17), dtype=object)
    arr[:, :] = u[0]
    lat.universes = arr
    lat.universes = [[u[1]] * 17] * 17
    assert lat.shape == (17, 17)
    with pytest.raises(UM2NotSupportedError):
        lat.pitch = (1.0, 1.0, 1.0)


def test_hex_lattice():
    u = [openmc.Universe() for _ in range(3)]
    lat = openmc.HexLattice()
    lat.center = (0.0, 0.0)
    lat.pitch = (1.25,)
    lat.universes = [[u[0]] * 12, [u[1]] * 6, [u[2]]]
    assert lat.num_rings == 3
    assert lat.get_universe((0, 0)) is u[2]
    assert lat.get_universe((1, 0)) is u[1]
    assert lat.get_universe((2, -1)) is u[0]
    assert '(2, 0)' in openmc.HexLattice.show_indices(3)
    with pytest.raises(ValueError):
        lat.universes = [[u[0]] * 11, [u[1]] * 6, [u[2]]]
    lat.orientation = 'x'
    idx, _ = lat.find_element((1.25, 0.0, 0.0))
    assert idx == (1, 0)
    assert 'HexLattice' in repr(lat)


def test_geometry_queries_and_find():
    fuel = openmc.Material(name='fuel')
    water = openmc.Material(name='water')
    cyl = openmc.ZCylinder(r=0.4)
    pin = openmc.Universe(cells=[openmc.Cell(fill=fuel, region=-cyl),
                                 openmc.Cell(fill=water, region=+cyl)])
    lat = openmc.RectLattice()
    lat.lower_left = (-1.0, -1.0)
    lat.pitch = (1.0, 1.0)
    lat.universes = [[pin, pin], [pin, pin]]
    box = openmc.model.RectangularPrism(2.0, 2.0, boundary_type='reflective')
    root = openmc.Cell(fill=lat, region=-box)
    geom = openmc.Geometry([root])
    assert set(m.name for m in geom.get_all_materials().values()) == {'fuel', 'water'}
    assert len(geom.get_all_lattices()) == 1
    assert len(geom.get_all_surfaces()) == 5
    path = geom.find((0.5, 0.5, 0.0))
    assert path[-1].fill is fuel
    assert geom.find((0.05, 0.05, 0.0))[-1].fill is water
    assert geom.get_materials_by_name('fue')[0] is fuel
    # translated universe
    cell = openmc.Cell(fill=pin, region=-openmc.ZCylinder(x0=3.0, r=1.0))
    cell.translation = (3.0, 0.0, 0.0)
    g2 = openmc.Geometry([cell])
    assert g2.find((3.1, 0.0, 0.0))[-1].fill is fuel


def test_geometry_xml(run_in_tmpdir):
    fuel = openmc.Material(name='fuel')
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cc', 10.0)
    cyl = openmc.ZCylinder(r=0.4)
    u = openmc.Universe(cells=[openmc.Cell(fill=fuel, region=-cyl),
                               openmc.Cell(region=+cyl)])
    lat = openmc.HexLattice()
    lat.center = (0., 0.)
    lat.pitch = (1.0,)
    lat.universes = [[u] * 6, [u]]
    lat.outer = u
    root = openmc.Cell(fill=lat, region=-openmc.ZCylinder(r=3.0, boundary_type='vacuum'))
    openmc.Geometry([root]).export_to_xml()
    text = open('geometry.xml').read()
    assert '<hex_lattice' in text and 'boundary="vacuum"' in text


def test_stubs_are_ignored():
    settings = openmc.Settings()
    settings.batches = 10
    settings.energy_mode = 'multi-group'
    settings.source = openmc.IndependentSource(space=openmc.stats.Box((-1, -1, -1), (1, 1, 1)))
    tally = openmc.Tally(name='flux')
    tally.filters = [openmc.MeshFilter(openmc.RegularMesh())]
    tally.scores = ['flux']
    tallies = openmc.Tallies([tally])
    model = openmc.Model(settings=settings, tallies=tallies)
    model.run()
    openmc.run()
    plot = openmc.Plot.from_geometry(model.geometry)
    plot.to_ipython_image()
    with pytest.raises(UM2NotSupportedError):
        openmc.StatePoint('statepoint.100.h5')
    with pytest.raises(UM2NotSupportedError):
        openmc.mgxs.Library(model.geometry).build_library()


def test_as_openmc():
    import sys
    import um2.openmc
    saved = {k: v for k, v in sys.modules.items()
             if k == 'openmc' or k.startswith('openmc.')}
    for k in saved:
        del sys.modules[k]
    try:
        um2.openmc.as_openmc()
        import openmc as o
        import openmc.mgxs
        import openmc.model
        from openmc.data import atomic_mass
        assert o is um2.openmc
        assert openmc.mgxs.EnergyGroups is um2.openmc.mgxs.EnergyGroups
        assert atomic_mass('H1') > 1.0
    finally:
        for k in [k for k in sys.modules if k == 'openmc' or k.startswith('openmc.')]:
            del sys.modules[k]
        sys.modules.update(saved)
