"""Convert real OpenMC models into UM2 models and meshes."""

import openmc
import um2.openmc as umc
from copy import deepcopy
from pathlib import Path

from .mpact import infer_hierarchy
from .pipeline import mesh_geometry
import math

SURFACE_REGISTRY = {
    openmc.XPlane: (umc.XPlane, ("x0",)),
    openmc.YPlane: (umc.YPlane, ("y0",)),
    openmc.ZPlane: (umc.ZPlane, ("z0",)),
    openmc.ZCylinder: (umc.ZCylinder, ("x0", "y0", "r")),
    openmc.Plane: (umc.Plane, ("a", "b", "c", "d")),
}

class OpenMCConverter:
    """Convert real OpenMC objects to UM2-compatible objects."""

    def __init__(self):
        self.surfaces = {}
        self.materials = {}
        self.cells = {}
        self.universes = {}
        self.lattices = {}

    def convert_surface(self, surface):
        if surface in self.surfaces:
            return self.surfaces[surface]

        entry = SURFACE_REGISTRY.get(type(surface))

        if entry is None:
            raise NotImplementedError(
                f"Unsupported OpenMC surface: {type(surface).__name__}"
            )

        target_class, attributes = entry

        kwargs = {
            attr: getattr(surface, attr)
            for attr in attributes
        }

        kwargs.update({
            "surface_id": surface.id,
            "name": surface.name,
            "boundary_type": surface.boundary_type,
            "albedo": surface.albedo,
        })

        converted = target_class(**kwargs)
        self.surfaces[surface] = converted

        return converted

    def convert_region(self, region):
        if region is None:
            return None

        if isinstance(region, openmc.Halfspace):
            surface = self.convert_surface(region.surface)
            return +surface if region.side == "+" else -surface

        if isinstance(region, openmc.Intersection):
            return umc.Intersection(
                [self.convert_region(r) for r in region]
            )

        if isinstance(region, openmc.Union):
            return umc.Union(
                [self.convert_region(r) for r in region]
            )

        if isinstance(region, openmc.Complement):
            return umc.Complement(
                self.convert_region(region.node)
            )

        raise NotImplementedError(
            f"Unsupported OpenMC region: {type(region).__name__}"
        )

    def convert_material(self, material):
        if material in self.materials:
            return self.materials[material]

        converted = umc.Material(
            material_id=material.id,
            name=material.name,
        )

        xs_name = f"material_{material.id}"
        converted.add_macroscopic(xs_name)

        self.materials[material] = converted
        return converted

    def convert_cell(self, cell):
        if cell in self.cells:
            return self.cells[cell]

        converted = umc.Cell(
            cell_id=cell.id,
            name=cell.name,
            region=self.convert_region(cell.region),
        )

        self.cells[cell] = converted

        if cell.fill_type == "material":
            converted.fill = self.convert_material(cell.fill)

        elif cell.fill_type == "universe":
            converted.fill = self.convert_universe(cell.fill)

        elif cell.fill_type == "lattice":
            converted.fill = self.convert_lattice(cell.fill)

        elif cell.fill_type != "void":
            raise NotImplementedError(
                f"Unsupported fill type: {cell.fill_type}"
            )

        if cell.translation is not None:
            converted.translation = cell.translation

        if cell.rotation is not None:
            converted.rotation = cell.rotation

        return converted

    def convert_universe(self, universe):
        if universe in self.universes:
            return self.universes[universe]

        if not isinstance(universe, openmc.Universe):
            raise NotImplementedError(
                f"Unsupported universe: {type(universe).__name__}"
            )

        converted = umc.Universe(
            universe_id=universe.id,
            name=universe.name,
        )

        self.universes[universe] = converted

        for cell in universe.cells.values():
            converted.add_cell(self.convert_cell(cell))

        return converted

    def convert_lattice(self, lattice):
        if lattice in self.lattices:
            return self.lattices[lattice]

        if not isinstance(lattice, openmc.RectLattice):
            raise NotImplementedError(
                f"Unsupported lattice type: {type(lattice).__name__}"
            )

        converted = umc.RectLattice(
            lattice_id=lattice.id,
            name=lattice.name,
        )

        self.lattices[lattice] = converted

        if lattice.ndim != 2:
            raise NotImplementedError(
                "UM2 only supports 2D rectangular lattices"
            )

        converted.pitch = lattice.pitch
        converted.lower_left = lattice.lower_left

        if lattice.outer is not None:
            converted.outer = self.convert_universe(lattice.outer)

        converted.universes = [
            [self.convert_universe(universe) for universe in row]
            for row in lattice.universes
        ]

        return converted

    def convert_geometry(self, geometry):
        root = self.convert_universe(geometry.root_universe)
        return umc.Geometry(root)

    def convert_model(self, model, mgxs_path):
        geometry = self.convert_geometry(model.geometry)

        materials = umc.Materials(
            [self.convert_material(m)
            for m in model.geometry.get_all_materials().values()]
        )

        materials.cross_sections = str(mgxs_path)

        return umc.Model(
            geometry=geometry,
            materials=materials,
        )


def generate_mgxs(
    model,
    energy_groups="CASMO-8",
    workdir="output/openmc",
    openmc_exec="openmc",
):
    """Run OpenMC and generate material-wise multigroup cross sections.

    Parameters
    ----------
    model : openmc.Model
        Real OpenMC continuous-energy model.
    energy_groups : str or openmc.mgxs.EnergyGroups
        Energy group structure.
    workdir : str or Path
        Directory for OpenMC outputs and the MGXS library.
    openmc_exec : str
        Path to the OpenMC executable.

    Returns
    -------
    Path
        Path to the generated mgxs.h5 file.
    """
    if not isinstance(model, openmc.Model):
        raise TypeError("Expected a real openmc.Model")

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Avoid changing the original model or its existing tallies.
    run_model = deepcopy(model)

    materials = sorted(
        run_model.geometry.get_all_materials().values(),
        key=lambda m: m.id,
    )

    if not materials:
        raise ValueError("OpenMC model contains no materials")

    if isinstance(energy_groups, str):
        groups = openmc.mgxs.EnergyGroups(energy_groups)
    else:
        groups = energy_groups

    # Create the MGXS library.
    mgxs_lib = openmc.mgxs.Library(run_model.geometry)

    mgxs_lib.energy_groups = groups
    mgxs_lib.domain_type = "material"
    mgxs_lib.domains = materials
    mgxs_lib.by_nuclide = False
    mgxs_lib.correction = None

    mgxs_lib.mgxs_types = [
        "total",
        "absorption",
        "fission",
        "nu-fission",
        "chi",
        "consistent scatter matrix",
        "consistent nu-scatter matrix",
    ]

    mgxs_lib.build_library()

    # Add MGXS tallies alongside any tallies already in the model.
    mgxs_lib.add_to_tallies(run_model.tallies)

    # Ensure the final statepoint and summary are generated.
    run_model.settings.statepoint = {
        "batches": [run_model.settings.batches]
    }

    output_settings = dict(run_model.settings.output or {})
    output_settings.pop("path", None)
    output_settings["summary"] = True
    run_model.settings.output = output_settings

    # Run continuous-energy OpenMC.
    statepoint_path = run_model.run(
        cwd=workdir,
        openmc_exec=openmc_exec,
    )

    if statepoint_path is None:
        raise RuntimeError("OpenMC did not produce a statepoint file")

    # Load tally results and link the geometry summary.
    with openmc.StatePoint(statepoint_path) as sp:
        if sp.summary is None:
            sp.link_with_summary(
                openmc.Summary(workdir / "summary.h5")
            )

        mgxs_lib.load_from_statepoint(sp)

    # Match the dataset names expected by convert_material().
    xs_names = [f"material_{mat.id}" for mat in materials]

    mg_library = mgxs_lib.create_mg_library(
        xs_type="macro",
        xsdata_names=xs_names,
    )

    mgxs_path = workdir / "mgxs.h5"
    mg_library.export_to_hdf5(str(mgxs_path))

    return mgxs_path

def validate_openmc_model(model):
    """Validate that the input is a real OpenMC model."""

    if not isinstance(model, openmc.Model):
        raise TypeError(
            f"Expected openmc.Model, got {type(model).__name__}"
        )

    if model.geometry is None or model.geometry.root_universe is None:
        raise ValueError("OpenMC model has no root universe")

    if not model.geometry.get_all_materials():
        raise ValueError("OpenMC model contains no materials")

def _infer_mpact_boundaries(geometry, plan):
    """Infer MPACT boundary conditions from OpenMC root-cell surfaces.

    Returns (west, north, east, south, top, bottom).
    """
    bc_codes = {
        "vacuum": 0,
        "reflective": 1,
        "periodic": 2,
    }

    xmin, ymin = plan.origin
    width, height = plan.size
    zmin, zmax = plan.z

    faces = {
        "west": (openmc.XPlane, "x0", xmin),
        "north": (openmc.YPlane, "y0", ymin + height),
        "east": (openmc.XPlane, "x0", xmin + width),
        "south": (openmc.YPlane, "y0", ymin),
        "top": (openmc.ZPlane, "z0", zmax),
        "bottom": (openmc.ZPlane, "z0", zmin),
    }

    matches = {face: set() for face in faces}

    # Only consider surfaces belonging to root-universe cells.
    # Internal pin/lattice surfaces should not define core BCs.
    for cell in geometry.root_universe.cells.values():
        if cell.region is None:
            continue

        for surface in cell.region.get_surfaces().values():
            for face, (surface_type, attr, coordinate) in faces.items():
                if isinstance(surface, surface_type) and math.isclose(
                    getattr(surface, attr),
                    coordinate,
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                ):
                    matches[face].add(surface.boundary_type)

    # An unbounded-z OpenMC model is treated as a 2D slice:
    # reflective top/bottom prevent artificial axial leakage.
    if not matches["top"] and not matches["bottom"]:
        matches["top"] = {"reflective"}
        matches["bottom"] = {"reflective"}

    result = {}

    for face, types in matches.items():
        if len(types) != 1:
            raise ValueError(
                f'Cannot uniquely determine the {face} boundary: '
                f'{sorted(types)}. Explicit boundary conditions are '
                'required for this geometry.'
            )

        boundary_type = next(iter(types))

        if boundary_type not in bc_codes:
            raise ValueError(
                f'Unsupported OpenMC {face} boundary: '
                f'{boundary_type!r}'
            )

        result[face] = bc_codes[boundary_type]

    # Periodic boundaries must have their corresponding partner.
    for first, second in (("west", "east"), ("north", "south"),
                          ("top", "bottom")):
        if (result[first] == 2) != (result[second] == 2):
            raise ValueError(
                f'Periodic {first}/{second} boundaries must be paired.'
            )

    return tuple(result[face] for face in (
        "west", "north", "east", "south", "top", "bottom"
    ))

def generate_um2_from_openmc(
    model,
    *,
    energy_groups="CASMO-8",
    kn_target=12.0,
    mesh_type="QuadraticTri",
    mgxs_path=None,
    workdir="output/openmc",
    openmc_exec="openmc",
):
    # 1. Validate model compatibility
    validate_openmc_model(model)

    # 2. Generate cross sections if not provided
    if mgxs_path is None:
        mgxs_path = generate_mgxs(
            model,
            energy_groups=energy_groups,
            workdir=workdir,
            openmc_exec=openmc_exec,
        )

    # 3. Convert the model
    converter = OpenMCConverter()
    um2_model = converter.convert_model(model, mgxs_path)

    # 4. Infer the MPACT hierarchy
    plan = infer_hierarchy(um2_model.geometry)

    # Extract the original OpenMC boundary conditions.
    boundary_conditions = _infer_mpact_boundaries(
        model.geometry, plan
    )

    # 5. Mesh without exporting
    result = mesh_geometry(
        um2_model,
        plan=plan,
        cross_sections=mgxs_path,
        kn_target=kn_target,
        mesh_type=mesh_type,
        output=None,
    )

    result.boundary_conditions = boundary_conditions
    return result