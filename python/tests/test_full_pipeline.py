
"""Integration tests for real OpenMC-to-UM2 conversion."""

import os
import shutil
from pathlib import Path

import numpy as np
import openmc
import pytest

import um2
from fixtures.openmc_pincell import create_model
from um2.generate_um2_from_openmc import generate_um2_from_openmc, generate_mgxs
from um2.pipeline import write_mesh_result


pytestmark = pytest.mark.skipif(
    os.environ.get("UM2_RUN_OPENMC_TESTS") != "1",
    reason="Set UM2_RUN_OPENMC_TESTS=1 to run OpenMC integration tests",
)


@pytest.fixture(scope="module")
def pincell_mgxs(tmp_path_factory):
    """Generate the pin-cell MGXS library once for both tests."""

    cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not cross_sections or not Path(cross_sections).is_file():
        pytest.skip("OPENMC_CROSS_SECTIONS is not configured")

    openmc_exec = os.environ.get("OPENMC_EXECUTABLE", "openmc")

    if not (
        Path(openmc_exec).is_file()
        or shutil.which(openmc_exec)
    ):
        pytest.skip("OpenMC executable not found")

    workdir = tmp_path_factory.mktemp("openmc_mgxs")
    model = create_model()

    mgxs_path = generate_mgxs(
        model,
        energy_groups="CASMO-8",
        workdir=workdir,
        openmc_exec=openmc_exec,
    )

    return model, mgxs_path


def test_generate_mgxs(pincell_mgxs):
    """Verify automatic material-wise MGXS generation."""

    model, mgxs_path = pincell_mgxs

    assert mgxs_path.is_file()

    xs = openmc.MGXSLibrary.from_hdf5(str(mgxs_path))

    assert xs.energy_groups.num_groups == 8
    assert len(xs.xsdatas) == 3

    expected_names = {
        f"material_{mat.id}"
        for mat in model.geometry.get_all_materials().values()
    }

    actual_names = {data.name for data in xs.xsdatas}

    assert actual_names == expected_names

    # Verify the required cross sections contain finite values.
    for data in xs.xsdatas:
        for name in (
            "total",
            "absorption",
            "scatter_matrix",
            "multiplicity_matrix",
        ):
            array = np.asarray(getattr(data, name))
            assert np.issubdtype(array.dtype, np.number)
            assert np.all(np.isfinite(array)), (
                f"{data.name}: invalid {name}"
            )

    # Verify that fuel has valid fission cross sections.
    fuel = next(
        data for data in xs.xsdatas
        if data.name == "material_1"
    )

    assert fuel.fission is not None
    assert np.all(np.isfinite(fuel.fission))
    assert np.any(np.asarray(fuel.fission) > 0)


def test_full_pipeline(pincell_mgxs, tmp_path):
    """Verify OpenMC geometry converts to UM2 and MPACT files."""

    pytest.importorskip("um2._um2")
    pytest.importorskip("gmsh")

    model, mgxs_path = pincell_mgxs

    output = tmp_path / "pincell.xdmf"

    result = generate_um2_from_openmc(
        model,
        mgxs_path=mgxs_path,
        kn_target=12.0,
        mesh_type="Tri",
    )

    # Confirm generation did not automatically export.
    assert not output.exists()

    files = write_mesh_result(result, output)

    # Verify that the exported MPACT input preserves the boundary conditions.
    input_text = output.with_suffix(".input").read_text()

    assert "bound_cond 1 1 1 1 1 1" in input_text
    assert "xslib USER pincell.xsl" in input_text
    assert "mat 4 0 :: Void" in input_text

    # Confirm explicit export succeeded.
    assert output in files

    # Verify all expected output files.
    expected_files = [
        "pincell.xdmf",
        "pincell.h5",
        "pincell.input",
        "pincell.volumes",
        "pincell.xs",
        "pincell.xsl",
    ]

    for filename in expected_files:
        path = tmp_path / filename
        assert path.is_file(), f"Missing output: {filename}"
        assert path.stat().st_size > 0

    # Verify the inferred MPACT hierarchy.
    assert len(result.plan.coarse_cells) == 1
    assert result.plan.coarse_cells[0].width == pytest.approx(1.26)
    assert result.plan.coarse_cells[0].height == pytest.approx(1.26)

    assert result.plan.z == pytest.approx((-1.0, 1.0))

    # Verify material names are preserved.
    assert set(result.materials) == {
        "uo2",
        "zirconium",
        "h2o",
        "Void",
    }

    assert result.material_xs_names == {
        "uo2": "material_1",
        "zirconium": "material_2",
        "h2o": "material_3",
    }

    assert result.cross_sections_path == Path(mgxs_path)

    xsl_path = tmp_path / "pincell.xsl"

    assert xsl_path.is_file()

    xsl = um2.UserXSLibrary(str(xsl_path))

    assert xsl.num_groups == 8
    assert set(xsl.names) == {
        "material_1",
        "material_2",
        "material_3",
        "Void",
    }

    void_xs = xsl.get_xsec("Void")

    for key in (
        "absorption",
        "fission",
        "nu_fission",
        "transport",
        "scatter_matrix",
    ):
        assert np.allclose(void_xs[key], 0.0)


def create_lattice_model():
    """Create a real OpenMC 2x2 lattice with repeated pin universes."""

    # Reuse the physical materials and simulation settings.
    model = create_model()

    uo2 = next(m for m in model.materials if m.name == "uo2")
    zirconium = next(
        m for m in model.materials if m.name == "zirconium"
    )
    water = next(m for m in model.materials if m.name == "h2o")

    # Build a fuel-pin universe with the original four regions.
    fuel_outer = openmc.ZCylinder(r=0.39)
    clad_inner = openmc.ZCylinder(r=0.40)
    clad_outer = openmc.ZCylinder(r=0.46)

    fuel_pin = openmc.Universe(
        cells=[
            openmc.Cell(
                fill=uo2,
                region=-fuel_outer,
            ),
            openmc.Cell(
                region=+fuel_outer & -clad_inner,
            ),
            openmc.Cell(
                fill=zirconium,
                region=+clad_inner & -clad_outer,
            ),
            openmc.Cell(
                fill=water,
                region=+clad_outer,
            ),
        ]
    )

    # A second universe containing only moderator.
    moderator_pin = openmc.Universe(
        cells=[openmc.Cell(fill=water)]
    )

    # OpenMC stores lattice rows from top to bottom.
    lattice = openmc.RectLattice()
    lattice.lower_left = (-1.26, -1.26)
    lattice.pitch = (1.26, 1.26)
    lattice.universes = [
        [fuel_pin, moderator_pin],
        [moderator_pin, fuel_pin],
    ]

    # Bound the full 2x2 lattice.
    left = openmc.XPlane(
        x0=-1.26, boundary_type="reflective"
    )
    right = openmc.XPlane(
        x0=1.26, boundary_type="reflective"
    )
    bottom = openmc.YPlane(
        y0=-1.26, boundary_type="reflective"
    )
    top = openmc.YPlane(
        y0=1.26, boundary_type="reflective"
    )

    z_bottom = openmc.ZPlane(
        z0=-1.0, boundary_type="reflective"
    )
    z_top = openmc.ZPlane(
        z0=1.0, boundary_type="reflective"
    )

    region = (
        +left & -right
        & +bottom & -top
        & +z_bottom & -z_top
    )

    root = openmc.Universe(
        cells=[
            openmc.Cell(fill=lattice, region=region)
        ]
    )

    model.geometry = openmc.Geometry(root)

    return model


def test_rect_lattice_pipeline(tmp_path):
    """Test real OpenMC 2x2 lattice -> MGXS -> UM2 -> MPACT."""

    pytest.importorskip("um2._um2")
    pytest.importorskip("gmsh")

    cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not cross_sections or not Path(cross_sections).is_file():
        pytest.skip("OPENMC_CROSS_SECTIONS is not configured")

    openmc_exec = os.environ.get("OPENMC_EXECUTABLE", "openmc")

    if not (
        Path(openmc_exec).is_file()
        or shutil.which(openmc_exec)
    ):
        pytest.skip("OpenMC executable not found")

    model = create_lattice_model()

    # Verify the real OpenMC geometry before converting.
    lattice = next(
        iter(model.geometry.get_all_lattices().values())
    )

    assert lattice.shape == (2, 2)
    assert lattice.universes[0, 0] is lattice.universes[1, 1]
    assert lattice.universes[0, 1] is lattice.universes[1, 0]

    # Generate MGXS for this lattice's neutron environment.
    mgxs_path = generate_mgxs(
        model,
        energy_groups="CASMO-8",
        workdir=tmp_path / "openmc",
        openmc_exec=openmc_exec,
    )

    assert mgxs_path.is_file()

    # Convert and mesh the real OpenMC lattice.
    output = tmp_path / "lattice.xdmf"

    # Convert and mesh the real OpenMC lattice.
    output = tmp_path / "lattice.xdmf"

    result = generate_um2_from_openmc(
        model,
        mgxs_path=mgxs_path,
        kn_target=12.0,
        mesh_type="Tri",
    )

    # Generation should not write the final files.
    assert not output.exists()

    # Explicitly export the generated model.
    files = write_mesh_result(result, output)

    assert output in files

    # Check output files.
    for suffix in (
        ".xdmf",
        ".h5",
        ".input",
        ".volumes",
        ".xs",
    ):
        path = output.with_suffix(suffix)

        assert path.is_file(), f"Missing {path.name}"
        assert path.stat().st_size > 0

    # A 2x2 pin lattice spans 2.52 x 2.52 cm.
    assert result.plan.z == pytest.approx((-1.0, 1.0))

    assert any(
        np.asarray(arr).shape == (2, 2)
        for arr in result.plan.rtms + result.plan.lattices
    ), "Expected a 2x2 structure in the inferred hierarchy"

    # Verify all physical materials appear in the result.
    assert {"uo2", "zirconium", "h2o", "Void"}.issubset(
        result.materials.keys()
    )

    print("2x2 rectangular lattice pipeline successful!")

def test_multiple_exports(pincell_mgxs, tmp_path):
    """A generated UM2 model can be written to multiple destinations."""
    pytest.importorskip("um2._um2")
    pytest.importorskip("gmsh")

    model, mgxs_path = pincell_mgxs

    result = generate_um2_from_openmc(
        model,
        mgxs_path=mgxs_path,
        kn_target=12.0,
        mesh_type="Tri",
    )

    first = tmp_path / "first.xdmf"
    second = tmp_path / "second.xdmf"

    # Generation should not export either file.
    assert not first.exists()
    assert not second.exists()

    write_mesh_result(result, first)
    write_mesh_result(result, second)

    # Both exports must contain the complete output set.
    for base in (first, second):
        for suffix in (".xdmf", ".h5", ".input", ".volumes", ".xs"):
            path = base.with_suffix(suffix)
            assert path.is_file()
            assert path.stat().st_size > 0

def test_mpact_boundary_inference():
    from types import SimpleNamespace
    import openmc
    from um2.generate_um2_from_openmc import _infer_mpact_boundaries

    west = openmc.XPlane(x0=-1, boundary_type="vacuum")
    east = openmc.XPlane(x0=1, boundary_type="reflective")
    south = openmc.YPlane(y0=-1, boundary_type="vacuum")
    north = openmc.YPlane(y0=1, boundary_type="reflective")

    cell = openmc.Cell(
        region=+west & -east & +south & -north
    )

    geometry = openmc.Geometry(
        openmc.Universe(cells=[cell])
    )

    plan = SimpleNamespace(
        origin=(-1, -1),
        size=(2, 2),
        z=(-0.5, 0.5),
    )

    assert _infer_mpact_boundaries(geometry, plan) == (
        0, 1, 1, 0, 1, 1
    )