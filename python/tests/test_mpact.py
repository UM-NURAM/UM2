"""Inference and construction of the MPACT spatial partition."""

import numpy as np
import pytest

import um2
import um2.openmc as openmc
from um2.mpact import Plan, grid_plan, infer_hierarchy
from conftest import requires_core


def pin(fuel, water, r=0.54):
    cyl = openmc.ZCylinder(r=r)
    return openmc.Universe(name=fuel.name, cells=[openmc.Cell(fill=fuel, region=-cyl),
                                                  openmc.Cell(fill=water, region=+cyl)])


@pytest.fixture
def c5g7():
    """The 2D C5G7 core (structure as in the mg-mode-part-i notebook)."""
    m = {n: openmc.Material(name=n) for n in ('uo2', 'mox', 'gt', 'water')}
    u = {n: pin(m[n], m['water']) for n in ('uo2', 'mox', 'gt')}
    u['water'] = openmc.Universe(name='water', cells=[openmc.Cell(fill=m['water'])])

    def assembly(fuel):
        lat = openmc.RectLattice(name=fuel)
        lat.lower_left = (-10.71, -10.71)
        lat.pitch = (1.26, 1.26)
        arr = np.empty((17, 17), dtype=object)
        arr[:, :] = u[fuel]
        arr[2, 5] = arr[8, 8] = u['gt']
        lat.universes = arr
        return openmc.Universe(cells=[openmc.Cell(fill=lat)])

    refl_lat = openmc.RectLattice()
    refl_lat.lower_left = (-10.71, -10.71)
    refl_lat.pitch = (21.42, 21.42)
    refl_lat.universes = [[u['water']]]
    refl = openmc.Universe(cells=[openmc.Cell(fill=refl_lat)])
    uo2, mox = assembly('uo2'), assembly('mox')
    core = openmc.RectLattice(name='core')
    core.lower_left = (-32.13, -32.13)
    core.pitch = (21.42, 21.42)
    core.universes = [[uo2, mox, refl], [mox, uo2, refl], [refl, refl, refl]]
    region = (+openmc.XPlane(-32.13, boundary_type='reflective') &
              -openmc.XPlane(32.13, boundary_type='vacuum') &
              +openmc.YPlane(-32.13, boundary_type='vacuum') &
              -openmc.YPlane(32.13, boundary_type='reflective'))
    root = openmc.Universe(cells=[openmc.Cell(fill=core, region=region)])
    return openmc.Geometry(root), core


def test_infer_c5g7(c5g7):
    geom, core = c5g7
    plan = infer_hierarchy(geom)
    assert plan.core.tolist() == [[0, 1, 2], [1, 0, 2], [2, 2, 2]]
    assert plan.origin == (-32.13, -32.13)
    assert plan.size == pytest.approx((64.26, 64.26))
    # uo2 pin, gt pin, mox pin, homogeneous water
    labels = sorted(c.label.split()[-1] for c in plan.coarse_cells)
    assert len(plan.coarse_cells) == 4
    assert labels == ['"gt"', '"mox"', '"uo2"', '"water"']
    assert all(c.width == pytest.approx(1.26) for c in plan.coarse_cells)
    # the reflector (a single 21.42 cm element) is subdivided into 17 x 17
    assert plan.lattices[2].shape == (17, 17) and len(np.unique(plan.lattices[2])) == 1
    # pin RTMs
    assert all(r.shape == (1, 1) for r in plan.rtms)
    assert 'core lattice' in str(plan)
    # Assembly lattices are stored top row first, as in OpenMC
    gt = next(i for i, c in enumerate(plan.coarse_cells) if '"gt"' in c.label)
    rtm_gt = next(i for i, r in enumerate(plan.rtms) if r[0, 0] == gt)
    assert plan.lattices[0][2, 5] == rtm_gt and plan.lattices[0][8, 8] == rtm_gt


def test_infer_assembly_rtm(c5g7):
    geom, _ = c5g7
    plan = infer_hierarchy(geom, rtm='assembly')
    assert all(l.shape == (1, 1) for l in plan.lattices)
    assert sorted(plan.rtms[i].shape for i in range(len(plan.rtms))) == [(17, 17)] * 3


def test_coarse_cell_boxes(c5g7):
    geom, _ = c5g7
    plan = infer_hierarchy(geom)
    boxes = plan.coarse_cell_boxes()
    assert len(boxes) == len(plan.coarse_cells)
    for cc, (x0, y0, x1, y1) in boxes.items():
        assert x1 - x0 == pytest.approx(1.26)
        # The universe at the box center is the coarse cell's universe
        path = geom.find(((x0 + x1) / 2 + 0.6, (y0 + y1) / 2, 0.0))
        label = plan.coarse_cells[cc].label
        assert any(f'"{u.name}"' in label for u in path if isinstance(u, openmc.Universe)
                   and u.name) or 'homogeneous' in label


def test_single_assembly_and_pin():
    fuel, water = openmc.Material(name='fuel'), openmc.Material(name='water')
    u = pin(fuel, water, r=0.4)
    lat = openmc.RectLattice()
    lat.lower_left = (0.0, 0.0)
    lat.pitch = (1.26, 1.26)
    lat.universes = [[u] * 17] * 17
    box = +openmc.XPlane(0.0) & -openmc.XPlane(21.42) & +openmc.YPlane(0.0) & \
        -openmc.YPlane(21.42) & +openmc.ZPlane(-10.0) & -openmc.ZPlane(10.0)
    plan = infer_hierarchy(openmc.Geometry([openmc.Cell(fill=lat, region=box)]))
    assert plan.core.shape == (1, 1) and len(plan.coarse_cells) == 1
    assert plan.z == (-10.0, 10.0)
    cyl_box = openmc.model.RectangularPrism(1.26, 1.26)
    plan = infer_hierarchy(openmc.Geometry(
        [openmc.Cell(fill=fuel, region=-openmc.ZCylinder(r=0.4)),
         openmc.Cell(fill=water, region=+openmc.ZCylinder(r=0.4) & -cyl_box)]))
    assert plan.core.shape == (1, 1) and len(plan.coarse_cells) == 1
    assert 'rectangular lattice' in plan.notes[0]


def test_grid_plan_lines():
    water = openmc.Material(name='water')
    x = [openmc.XPlane(v) for v in (0.0, 1.0, 3.0)]
    y = [openmc.YPlane(v) for v in (0.0, 2.0)]
    geom = openmc.Geometry([
        openmc.Cell(fill=water, region=+x[0] & -x[1] & +y[0] & -y[1]),
        openmc.Cell(fill=water, region=+x[1] & -x[2] & +y[0] & -y[1])])
    plan = grid_plan(geom, coarse_cell_size=1.0)
    widths = [plan.coarse_cells[i].width for i in plan.rtms[0][0]]
    assert widths == pytest.approx([1.0, 1.0, 1.0])
    assert plan.rtms[0].shape == (2, 3)
    plan2 = grid_plan(geom, x=[0.0, 3.0], y=[0.0, 2.0])
    assert len(plan2.coarse_cells) == 1


def test_manual_plan_and_validation():
    plan = Plan()
    a = plan.add_coarse_cell(1.0, 1.0, 'a')
    b = plan.add_coarse_cell(2.0, 1.0, 'b')
    r1 = plan.add_rtm([[a, a]])
    r2 = plan.add_rtm([[b]])
    lat = plan.add_lattice([[r1, r2]])
    asy = plan.add_assembly([lat])
    plan.set_core([[asy]])
    plan.validate()
    assert plan.size == (4.0, 1.0)
    with pytest.raises(ValueError):
        plan.add_rtm([[a], [b]])
        plan.validate()
    bad = Plan()
    c = bad.add_coarse_cell(1.0, 1.0)
    d = bad.add_coarse_cell(1.0, 2.0)
    bad.set_core([[bad.add_assembly([bad.add_lattice([[bad.add_rtm([[c]]),
                                                        bad.add_rtm([[d]])]])])]])
    with pytest.raises(ValueError, match='same size'):
        bad.validate()
    with pytest.raises(ValueError):
        Plan().add_rtm([[0]])


def test_rtm_size_mismatch_raises():
    water = openmc.Material(name='water')
    u = openmc.Universe(cells=[openmc.Cell(fill=water)])
    small = openmc.RectLattice()
    small.lower_left = (-1.0, -1.0)
    small.pitch = (1.0, 1.0)
    small.universes = [[u, u], [u, u]]
    su = openmc.Universe(cells=[openmc.Cell(fill=small)])
    other = openmc.RectLattice()
    other.lower_left = (-1.0, -1.0)
    other.pitch = (0.7, 0.7)   # 2/0.7 is not an integer
    other.universes = [[u]]
    ou = openmc.Universe(cells=[openmc.Cell(fill=water)])
    core = openmc.RectLattice()
    core.lower_left = (0.0, 0.0)
    core.pitch = (2.0, 2.0)
    core.universes = [[su, ou]]
    box = openmc.model.RectangularPrism(4.0, 2.0, origin=(2.0, 1.0))
    geom = openmc.Geometry([openmc.Cell(fill=core, region=-box)])
    plan = infer_hierarchy(geom)     # ou is subdivided into 1 x 1 cm cells
    assert plan.lattices[1].shape == (2, 2)
    with pytest.raises(ValueError, match='multiple of the RTM size'):
        infer_hierarchy(geom, rtm_size=(0.7, 0.7))


@requires_core
def test_plan_to_um2(c5g7):
    geom, _ = c5g7
    um2.initialize(log_level=1)
    plan = infer_hierarchy(geom)
    model = plan.to_um2()
    assert model.num_coarse_cells == len(plan.coarse_cells)
    assert model.num_coarse_cells_total == 51 * 51
    assert model.num_assemblies == 3
    # UM2 stores the core bottom row first
    assert model.core_children() == plan.core[::-1].tolist()
