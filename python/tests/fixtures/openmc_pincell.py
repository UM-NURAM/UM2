
import openmc


def create_model():
    # Materials from the original pincell notebook
    uo2 = openmc.Material(name="uo2")
    uo2.add_nuclide("U235", 0.03)
    uo2.add_nuclide("U238", 0.97)
    uo2.add_nuclide("O16", 2.0)
    uo2.set_density("g/cm3", 10.0)

    zirconium = openmc.Material(name="zirconium")
    zirconium.add_element("Zr", 1.0)
    zirconium.set_density("g/cm3", 6.6)

    water = openmc.Material(name="h2o")
    water.add_nuclide("H1", 2.0)
    water.add_element("O", 1.0)
    water.set_density("g/cm3", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")

    # Geometry from the original notebook
    fuel_outer = openmc.ZCylinder(r=0.39)
    clad_inner = openmc.ZCylinder(r=0.40)
    clad_outer = openmc.ZCylinder(r=0.46)

    pitch = 1.26
    left = openmc.XPlane(-pitch / 2, boundary_type="reflective")
    right = openmc.XPlane(pitch / 2, boundary_type="reflective")
    bottom = openmc.YPlane(-pitch / 2, boundary_type="reflective")
    top = openmc.YPlane(pitch / 2, boundary_type="reflective")

    square = +left & -right & +bottom & -top

    z_bottom = openmc.ZPlane(z0=-1.0, boundary_type="reflective")
    z_top = openmc.ZPlane(z0=1.0, boundary_type="reflective")

    axial = +z_bottom & -z_top

    fuel = openmc.Cell(
        name="fuel", fill=uo2, region=-fuel_outer & square & axial,
    )
    gap = openmc.Cell(
        name="air gap",
        region=+fuel_outer & -clad_inner & square & axial,
    )
    clad = openmc.Cell(
        name="clad",
        fill=zirconium,
        region=+clad_inner & -clad_outer & square & axial,
    )
    moderator = openmc.Cell(
        name="moderator",
        fill=water,
        region=+clad_outer & square & axial,
    )

    geometry = openmc.Geometry(
        openmc.Universe(cells=[fuel, gap, clad, moderator])
    )

    model = openmc.Model()
    model.materials = openmc.Materials(
        [uo2, zirconium, water]
    )
    model.geometry = geometry

    model.settings.run_mode = "eigenvalue"

    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-0.5, -0.5, -0.9),
            (0.5, 0.5, 0.9),
        ),
        constraints={"fissionable": True},
    )

    model.settings.batches = 100
    model.settings.inactive = 10
    model.settings.particles = 1000

    return model
