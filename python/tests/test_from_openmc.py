"""Tests for converting real OpenMC geometry into UM2 geometry."""

import openmc
import pytest

from um2.generate_um2_from_openmc import OpenMCConverter

@pytest.fixture(autouse=True)
def reset_real_openmc_ids():
    openmc.reset_auto_ids()
    yield

def test_convert_pincell_geometry():
    fuel = openmc.Material(material_id=101, name="fuel")
    water = openmc.Material(material_id=102, name="water")

    fuel_outer = openmc.ZCylinder(surface_id=201, r=0.39)

    left = openmc.XPlane(surface_id=202, x0=-0.63)
    right = openmc.XPlane(surface_id=203, x0=0.63)
    bottom = openmc.YPlane(surface_id=204, y0=-0.63)
    top = openmc.YPlane(surface_id=205, y0=0.63)

    square = +left & -right & +bottom & -top

    fuel_cell = openmc.Cell(
        cell_id=301,
        fill=fuel,
        region=-fuel_outer & square,
    )

    water_cell = openmc.Cell(
        cell_id=302,
        fill=water,
        region=+fuel_outer & square,
    )

    root = openmc.Universe(
        universe_id=401,
        cells=[fuel_cell, water_cell],
    )

    geometry = openmc.Geometry(root)

    converter = OpenMCConverter()
    converted = converter.convert_geometry(geometry)

    assert len(converted.get_all_cells()) == 2
    assert len(converted.get_all_materials()) == 2
    assert len(converted.get_all_surfaces()) == 5

    bb = converted.bounding_box
    assert bb.lower_left[:2] == pytest.approx((-0.63, -0.63))
    assert bb.upper_right[:2] == pytest.approx((0.63, 0.63))

    center = converted.find((0.0, 0.0, 0.0))
    outside = converted.find((0.5, 0.5, 0.0))

    assert center[-1].fill.name == "fuel"
    assert outside[-1].fill.name == "water"

    assert converted.root_universe.id == 401
    assert set(converted.get_all_cells()) == {301, 302}
    assert set(converted.get_all_materials()) == {101, 102}

def test_convert_full_pincell():
    fuel = openmc.Material(material_id=101, name="uo2")
    clad = openmc.Material(material_id=102, name="zirconium")
    water = openmc.Material(material_id=103, name="h2o")

    fuel_outer = openmc.ZCylinder(r=0.39)
    clad_inner = openmc.ZCylinder(r=0.40)
    clad_outer = openmc.ZCylinder(r=0.46)

    left = openmc.XPlane(x0=-0.63)
    right = openmc.XPlane(x0=0.63)
    bottom = openmc.YPlane(y0=-0.63)
    top = openmc.YPlane(y0=0.63)

    square = +left & -right & +bottom & -top

    cells = [
        openmc.Cell(fill=fuel, region=-fuel_outer & square),
        openmc.Cell(region=+fuel_outer & -clad_inner & square),
        openmc.Cell(fill=clad, region=+clad_inner & -clad_outer & square),
        openmc.Cell(fill=water, region=+clad_outer & square),
    ]

    geometry = openmc.Geometry(openmc.Universe(cells=cells))

    converter = OpenMCConverter()
    converted = converter.convert_geometry(geometry)

    assert len(converted.get_all_cells()) == 4
    assert len(converted.get_all_materials()) == 3
    assert len(converted.get_all_surfaces()) == 7

    # Verify which material occupies each position
    def material_at(x, y):
        cell = converted.find((x, y, 0.0))[-1]
        return cell.fill.name if cell.fill is not None else None

    assert material_at(0.0, 0.0) == "uo2"
    assert material_at(0.395, 0.0) is None
    assert material_at(0.43, 0.0) == "zirconium"
    assert material_at(0.55, 0.0) == "h2o"

def test_convert_rect_lattice():
    fuel = openmc.Material(material_id=101, name="fuel")
    water = openmc.Material(material_id=102, name="water")

    # Create two universes with different materials
    fuel_universe = openmc.Universe(
        universe_id=201,
        cells=[openmc.Cell(cell_id=301, fill=fuel)],
    )

    water_universe = openmc.Universe(
        universe_id=202,
        cells=[openmc.Cell(cell_id=302, fill=water)],
    )

    # 2x2 lattice, with the fuel universe repeated
    lattice = openmc.RectLattice(lattice_id=401)
    lattice.lower_left = (-1.0, -1.0)
    lattice.pitch = (1.0, 1.0)
    lattice.universes = [
        [fuel_universe, water_universe],
        [water_universe, fuel_universe],
    ]

    # Bound the lattice within a 2x2 square
    left = openmc.XPlane(x0=-1.0)
    right = openmc.XPlane(x0=1.0)
    bottom = openmc.YPlane(y0=-1.0)
    top = openmc.YPlane(y0=1.0)

    square = +left & -right & +bottom & -top

    root = openmc.Universe(
        universe_id=203,
        cells=[
            openmc.Cell(
                cell_id=303,
                fill=lattice,
                region=square,
            )
        ],
    )

    geometry = openmc.Geometry(root)

    converter = OpenMCConverter()
    converted = converter.convert_geometry(geometry)

    # Verify the lattice's properties
    converted_lattice = converted.root_universe.cells[303].fill

    assert converted_lattice.id == 401
    assert converted_lattice.shape == (2, 2)
    assert converted_lattice.pitch == pytest.approx((1.0, 1.0))
    assert converted_lattice.lower_left == pytest.approx((-1.0, -1.0))

    # Check universe arrangement (row 0 is the top row)
    assert converted_lattice.universes[0, 0].id == 201
    assert converted_lattice.universes[0, 1].id == 202
    assert converted_lattice.universes[1, 0].id == 202
    assert converted_lattice.universes[1, 1].id == 201

    # Repeated universes should be the SAME objects
    assert (
        converted_lattice.universes[0, 0]
        is converted_lattice.universes[1, 1]
    )
    assert (
        converted_lattice.universes[0, 1]
        is converted_lattice.universes[1, 0]
    )

    # Verify material assignments at different lattice positions
    assert converted.find((-0.5, 0.5, 0.0))[-1].fill.name == "fuel"
    assert converted.find((0.5, 0.5, 0.0))[-1].fill.name == "water"
    assert converted.find((-0.5, -0.5, 0.0))[-1].fill.name == "water"
    assert converted.find((0.5, -0.5, 0.0))[-1].fill.name == "fuel"

    # Check total number of unique materials
    assert len(converted.get_all_materials()) == 2

def test_convert_axial_planes():
    converter = OpenMCConverter()

    bottom = openmc.ZPlane(
        z0=-0.5,
        boundary_type="reflective",
    )
    top = openmc.ZPlane(
        z0=0.5,
        boundary_type="reflective",
    )

    converted_bottom = converter.convert_surface(bottom)
    converted_top = converter.convert_surface(top)

    assert converted_bottom.z0 == pytest.approx(-0.5)
    assert converted_top.z0 == pytest.approx(0.5)
    assert converted_bottom.boundary_type == "reflective"
    assert converted_top.boundary_type == "reflective"