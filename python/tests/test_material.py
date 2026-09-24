"""Tests of um2.openmc materials against reference values."""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import um2.openmc as openmc
from um2.openmc import data

N_A = data.AVOGADRO


def test_water_atom_densities():
    water = openmc.Material(name='h2o')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)
    m_h, m_o = data.atomic_mass('H1'), data.atomic_mass('O16')
    n = 1.0 / (2*m_h + m_o) * N_A * 1e-24     # molecules / b-cm
    dens = water.get_nuclide_atom_densities()
    assert dens['H1'] == pytest.approx(2*n)
    assert dens['O16'] == pytest.approx(n)
    assert water.get_mass_density() == pytest.approx(1.0)
    assert water.average_molar_mass == pytest.approx((2*m_h + m_o) / 3)


@pytest.mark.parametrize('units, value, factor', [
    ('atom/b-cm', 0.1, 1.0), ('atom/cm3', 0.1e24, 1.0),
])
def test_atom_density_units(units, value, factor):
    m = openmc.Material()
    m.add_nuclide('U235', 1.0)
    m.add_nuclide('U238', 3.0)
    m.set_density(units, value)
    dens = m.get_nuclide_atom_densities()
    assert sum(dens.values()) == pytest.approx(0.1 * factor)
    assert dens['U238'] / dens['U235'] == pytest.approx(3.0)


def test_kg_m3_and_sum():
    a = openmc.Material()
    a.add_nuclide('O16', 1.0)
    a.set_density('kg/m3', 1000.0)
    b = openmc.Material()
    b.add_nuclide('O16', 1.0)
    b.set_density('g/cc', 1.0)
    assert a.get_nuclide_atom_densities()['O16'] == \
        pytest.approx(b.get_nuclide_atom_densities()['O16'])
    s = openmc.Material()
    s.add_nuclide('H1', 0.02)
    s.add_nuclide('O16', 0.01)
    s.set_density('sum')
    assert s.get_nuclide_atom_densities() == pytest.approx({'H1': 0.02, 'O16': 0.01})


def test_weight_percent():
    m = openmc.Material()
    m.add_nuclide('U235', 0.05, 'wo')
    m.add_nuclide('U238', 0.95, 'wo')
    m.set_density('g/cc', 10.0)
    dens = m.get_nuclide_atom_densities()
    mass = {n: d * data.atomic_mass(n) for n, d in dens.items()}
    assert mass['U235'] / (mass['U235'] + mass['U238']) == pytest.approx(0.05)
    assert m.get_mass_density() == pytest.approx(10.0)


def test_add_element_natural():
    zr = openmc.Material()
    zr.add_element('Zr', 1.0)
    assert zr.get_nuclides() == ['Zr90', 'Zr91', 'Zr92', 'Zr94', 'Zr96']
    total = sum(p.percent for p in zr.nuclides)
    assert total == pytest.approx(1.0)
    # element names are accepted too
    zr2 = openmc.Material()
    zr2.add_element('zirconium', 1.0)
    assert zr2.get_nuclides() == zr.get_nuclides()
    with pytest.raises(ValueError):
        zr.add_element('zr', 1.0)
    with pytest.raises(ValueError):
        zr.add_element('Xx', 1.0)


def test_uranium_enrichment():
    u = openmc.Material()
    u.add_element('U', 1.0, enrichment=3.0)
    u.set_density('g/cc', 10.0)
    dens = u.get_nuclide_atom_densities()
    mass = {n: d * data.atomic_mass(n) for n, d in dens.items()}
    total = sum(mass.values())
    assert mass['U235'] / total == pytest.approx(0.03)
    assert mass['U234'] / total == pytest.approx(0.0089 * 0.03)
    assert mass['U236'] / total == pytest.approx(0.0046 * 0.03)
    with pytest.raises(ValueError):
        openmc.Material().add_element('Zr', 1.0, enrichment=3.0)


def test_two_isotope_enrichment():
    b = openmc.Material()
    b.add_element('B', 1.0, enrichment=90.0, enrichment_target='B10',
                  enrichment_type='ao')
    fracs = {n.name: n.percent for n in b.nuclides}
    assert fracs['B10'] == pytest.approx(0.9)
    assert fracs['B11'] == pytest.approx(0.1)


def test_elements_from_formula():
    m = openmc.Material()
    m.add_elements_from_formula('H2O')
    elements = m.get_element_atom_densities if False else None  # noqa: F841
    h = sum(n.percent for n in m.nuclides if n.name.startswith('H'))
    o = sum(n.percent for n in m.nuclides if n.name.startswith('O'))
    assert h == pytest.approx(2/3)
    assert o == pytest.approx(1/3)
    m2 = openmc.Material()
    m2.add_elements_from_formula('(NH4)2SO4')
    n = sum(x.percent for x in m2.nuclides if x.name.startswith('N'))
    assert n == pytest.approx(2/15)
    with pytest.raises(ValueError):
        m2.add_elements_from_formula('H2.5O')


def test_mix_materials():
    uo2 = openmc.Material(name='uo2')
    uo2.add_nuclide('U235', 0.03)
    uo2.add_nuclide('U238', 0.97)
    uo2.add_nuclide('O16', 2.0)
    uo2.set_density('g/cm3', 10.0)
    puo2 = openmc.Material(name='puo2')
    puo2.add_nuclide('Pu239', 0.94)
    puo2.add_nuclide('Pu240', 0.06)
    puo2.add_nuclide('O16', 2.0)
    puo2.set_density('g/cm3', 11.5)
    mox = openmc.Material.mix_materials([uo2, puo2], [0.97, 0.03], 'wo')
    assert mox.name == 'uo2(0.97)-puo2(0.03)'
    # Mass of Pu in the mixture: 3% of the mass comes from PuO2
    dens = mox.get_nuclide_atom_densities()
    mass = {n: d * data.atomic_mass(n) for n, d in dens.items()}
    pu_mass = mass['Pu239'] + mass['Pu240']
    puo2_mass_frac = sum(puo2.get_mass_density(n) for n in ('Pu239', 'Pu240')) \
        / puo2.get_mass_density()
    assert pu_mass / sum(mass.values()) == pytest.approx(0.03 * puo2_mass_frac)
    # Volume fractions
    vol = openmc.Material.mix_materials([uo2, puo2], [0.5, 0.5], 'vo')
    assert vol.get_mass_density() == pytest.approx(0.5 * (10.0 + 11.5))
    with pytest.raises(ValueError):
        openmc.Material.mix_materials([uo2, puo2], [0.5, 0.4], 'ao')


def test_macroscopic():
    m = openmc.Material(name='uo2')
    m.add_macroscopic('UO2')
    assert m.density == 1.0 and m.density_units == 'macro'
    assert m.macroscopic == 'UO2'
    with pytest.raises(ValueError):
        m.add_macroscopic('other')
    with pytest.raises(ValueError):
        m.add_nuclide('U235', 1.0)
    n = openmc.Material()
    n.add_nuclide('H1', 1.0)
    with pytest.raises(ValueError):
        n.add_macroscopic('UO2')


def test_ids():
    a = openmc.Material()
    b = openmc.Material()
    assert b.id == a.id + 1
    c = openmc.Material(material_id=100)
    assert c.id == 100
    with pytest.warns(openmc.IDWarning):
        openmc.Material(material_id=100)
    openmc.reset_auto_ids()
    assert openmc.Material().id == 1


def test_remove_and_clone():
    m = openmc.Material(name='w')
    m.add_nuclide('H1', 2.0)
    m.add_nuclide('O16', 1.0)
    m.remove_nuclide('O16')
    m.add_element('O', 1.0)
    assert 'O16' in m.get_nuclides() and 'O17' in m.get_nuclides()
    m.remove_element('O')
    assert m.get_nuclides() == ['H1']
    m.set_density('g/cc', 1.0)
    c = m.clone()
    assert c.id != m.id and c.get_nuclides() == ['H1']


def test_materials_collection_and_xml(run_in_tmpdir):
    uo2 = openmc.Material(1, 'uo2')
    uo2.add_nuclide('U235', 0.03)
    uo2.set_density('g/cm3', 10.0)
    water = openmc.Material(name='h2o')
    water.add_nuclide('H1', 2.0)
    water.set_density('g/cm3', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')
    fuel = openmc.Material(name='mg')
    fuel.add_macroscopic('uo2')
    mats = openmc.Materials([uo2])
    mats.append(water)
    mats += [fuel]
    assert isinstance(mats, list) and len(mats) == 3
    with pytest.raises(TypeError):
        mats.append('not a material')
    mats.cross_sections = 'mgxs.h5'
    mats.export_to_xml()
    root = ET.parse('materials.xml').getroot()
    assert root.find('cross_sections').text.endswith('mgxs.h5')
    assert [m.get('name') for m in root.findall('material')] == ['uo2', 'h2o', 'mg']
    read = openmc.Materials.from_xml('materials.xml')
    assert read[2].macroscopic == 'uo2'
    assert read[1]._sab == [('c_H_in_H2O', 1.0)]
    assert read[0].get_nuclide_atom_densities() == \
        pytest.approx(uo2.get_nuclide_atom_densities())


def test_zam_and_masses():
    assert data.zam('U235') == (92, 235, 0)
    assert data.zam('Am242_m1') == (95, 242, 1)
    with pytest.raises(ValueError):
        data.zam('U-235')
    assert data.atomic_mass('H1') == pytest.approx(1.00782503, rel=1e-8)
    assert data.atomic_weight('C') == pytest.approx(12.011, rel=1e-3)
    assert data.ELEMENT_SYMBOL['uranium'] == 'U'
