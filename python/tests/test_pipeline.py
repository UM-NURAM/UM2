"""The C++ bindings and the mesh/export pipeline."""

import math

import numpy as np
import pytest

import um2
import um2.openmc as openmc
from conftest import C5G7_XSL, requires_core

pytestmark = requires_core


def test_shared_gmsh_model(gmsh_session):
    """UM2 (C++) and the gmsh Python API operate on the same gmsh model."""
    gmsh = gmsh_session
    core = um2._core()
    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    assert core.gmsh.count_entities(2) == len(gmsh.model.getEntities(2)) == 1
    tag = gmsh.model.addPhysicalGroup(2, [1], name='Material_water')
    gmsh.model.setColor([(2, 1)], 10, 20, 30)
    mats = core.gmsh.get_materials()
    assert [m.name for m in mats] == ['water']
    assert mats[0].color[:3] == (10, 20, 30)
    assert tag > 0


def test_errors_raise_exceptions(gmsh_session):
    core = um2._core()
    model = core.mpact.Model()
    with pytest.raises(um2.UM2Error, match='Coarse cell 3 does not exist'):
        model.add_rtm([[3]])
    with pytest.raises(um2.UM2Error):
        model.add_material(um2.Material('no_xs'))       # validation requires XS
    model.add_material(um2.Material('no_xs'), validate=False)
    # UM2 keeps working (the process did not exit)
    assert model.add_coarse_cell((1.0, 1.0)) == 0


def test_material_xsec():
    m = um2.Material('fuel')
    ss = np.array([[0.2, 0.0], [0.09, 0.4]])
    m.set_xsec(np.array([0.01, 0.1]), np.array([0.005, 0.05]),
               np.array([0.0125, 0.125]), np.array([0.3, 0.5]), ss)
    d = m.get_xsec()
    assert m.has_xsec() and d['fissile'] and m.num_groups == 2
    assert np.allclose(d['scatter'], [0.29, 0.4])
    assert np.allclose(d['scatter_matrix'], ss)
    with pytest.raises(ValueError):
        m.set_xsec(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros((3, 3)))
    m.color = (1, 2, 3)
    assert m.color == (1, 2, 3, 255)


def pin_model(cross_sections):
    names = ['UO2-3.3', 'Moderator']
    mats = {}
    for n in names:
        m = openmc.Material(name=n)
        m.add_macroscopic(n)
        mats[n] = m
    materials = openmc.Materials(mats.values())
    materials.cross_sections = cross_sections
    fuel = openmc.ZCylinder(r=0.54)
    box = openmc.model.RectangularPrism(1.26, 1.26, boundary_type='reflective')
    root = openmc.Universe(cells=[openmc.Cell(fill=mats['UO2-3.3'], region=-fuel),
                                  openmc.Cell(fill=mats['Moderator'], region=+fuel & -box)])
    return openmc.Model(geometry=openmc.Geometry(root), materials=materials)


@pytest.mark.parametrize('mesh_type', ['Tri', 'QuadraticTri', 'Quad'])
def test_pin_cell_pipeline(gmsh_session, tmp_path, mesh_type):
    model = pin_model(C5G7_XSL)
    out = tmp_path / 'pin.xdmf'
    # gmsh's all-quad meshes can have inverted elements if they are very coarse
    kn = 20.0 if mesh_type == 'Quad' else 8.0
    result = um2.mesh_geometry(model, mesh_type=mesh_type, kn_target=kn, output=out,
                               write_knudsen_data=True, write_xsec_data=True)
    assert out.exists() and (tmp_path / 'pin.h5').exists()
    assert result.model.num_fine_cells_total > 20
    volumes = dict(line.split() for line in (tmp_path / 'pin.volumes').read_text().splitlines())
    # Linear elements approximate the circle by a polygon
    rtol = 5e-2 if mesh_type in ('Tri', 'Quad') else 5e-4
    assert float(volumes['UO2-3.3']) == pytest.approx(math.pi * 0.54**2, rel=rtol)
    assert float(volumes['UO2-3.3']) + float(volumes['Moderator']) == \
        pytest.approx(1.26**2, rel=1e-9)
    # The cross sections given to UM2 are the converted MPACT data
    fuel = result.materials['UO2-3.3'].get_xsec()
    assert fuel['transport'][0] == pytest.approx(1.77949E-1, rel=1e-4)
    # The MPACT model can be read back
    core = um2._core()
    m2 = core.mpact.Model()
    m2.read(str(out))
    assert m2.num_fine_cells_total == result.model.num_fine_cells_total


def test_pipeline_without_cross_sections(gmsh_session):
    fuel, water = openmc.Material(name='fuel'), openmc.Material(name='water')
    for m in (fuel, water):
        m.add_nuclide('H1', 1.0)
        m.set_density('g/cc', 1.0)
    cyl = openmc.ZCylinder(r=0.5)
    box = openmc.model.RectangularPrism(1.26, 1.26)
    geom = openmc.Geometry([openmc.Cell(fill=fuel, region=-cyl),
                            openmc.Cell(fill=water, region=+cyl & -box)])
    result = um2.mesh_geometry(geom, mesh_size=0.1, mesh_type='Tri')
    assert not result.materials['fuel'].has_xsec()
    areas = um2.material_areas()
    assert areas['fuel'] == pytest.approx(math.pi * 0.25, rel=1e-9)
    with pytest.raises(ValueError, match='kn_target requires cross sections'):
        um2.mesh_geometry(geom, kn_target=4.0)


def test_inverted_elements_warning(gmsh_session, monkeypatch):
    from um2.pipeline import _check_mesh_quality
    gmsh = gmsh_session
    um2.mesh_geometry(pin_model(C5G7_XSL), mesh_type='Tri', kn_target=8.0)
    assert _check_mesh_quality() == 0
    # gmsh meshes are not always valid (e.g. coarse all-quad meshes)
    monkeypatch.setattr(gmsh.model.mesh, 'getElementQualities',
                        lambda tags, name: -np.ones(len(tags)))
    with pytest.warns(UserWarning, match='inverted'):
        assert _check_mesh_quality() > 0


def test_void_and_outside(gmsh_session, tmp_path):
    """Void cells and space outside a hexagonal boundary get zero cross sections."""
    model = pin_model(C5G7_XSL)
    root = model.geometry.root_universe
    fuel_cell, mod_cell = root.cells.values()
    gap = openmc.ZCylinder(r=0.6)
    hexagon = openmc.model.HexagonalPrism(edge_length=1.0, boundary_type='vacuum')
    mod_cell.region = +gap & -hexagon
    root.add_cell(openmc.Cell(region=+fuel_cell.region.surface & -gap))
    with pytest.warns(UserWarning, match="filled with 'Void'"):
        result = um2.mesh_geometry(model, kn_target=8.0, output=tmp_path / 'hex.xdmf')
    void = result.materials['Void'].get_xsec()
    assert np.all(void['transport'] == 0)
    areas = um2.material_areas()
    box = result.plan.size[0] * result.plan.size[1]
    assert areas['Void'] == pytest.approx(math.pi * (0.36 - 0.54**2) + box
                                          - 1.5 * math.sqrt(3), rel=1e-6)


def test_missing_macroscopic_raises(gmsh_session):
    model = pin_model(C5G7_XSL)
    next(iter(model.materials)).remove_macroscopic('UO2-3.3')
    next(iter(model.materials)).add_macroscopic('does-not-exist')
    with pytest.raises(KeyError, match='does-not-exist'):
        um2.mesh_geometry(model)


def test_c5g7_core(gmsh_session, tmp_path):
    """The full 2D C5G7 core: only the unique pins are built and meshed."""
    from test_mpact import c5g7 as c5g7_fixture
    geom, _ = c5g7_fixture.__wrapped__()
    library = um2.xs.read_xsl(C5G7_XSL)
    for mat in geom.get_all_materials().values():
        name = {'uo2': 'UO2-3.3', 'mox': 'MOX-4.3', 'gt': 'GuideTube',
                'water': 'Moderator'}[mat.name]
        mat.add_macroscopic(name)
    result = um2.mesh_geometry(geom, library=library, kn_target=12.0,
                               output=tmp_path / 'c5g7.xdmf')
    assert result.model.num_coarse_cells == 4
    assert result.model.num_coarse_cells_total == 51 * 51
    volumes = dict(l.split() for l in (tmp_path / 'c5g7.volumes').read_text().splitlines())
    pin_area = math.pi * 0.54**2
    assert float(volumes['gt']) == pytest.approx(4 * 2 * pin_area, rel=1e-4)
    assert float(volumes['uo2']) == pytest.approx(2 * 287 * pin_area, rel=1e-4)
    assert sum(map(float, volumes.values())) == pytest.approx(64.26**2, rel=1e-9)
