"""Multigroup cross sections: OpenMC XSdata/MGXSLibrary and MPACT .xsl."""

import numpy as np
import pytest

import um2.openmc as openmc
from um2 import xs
from conftest import C5G7_XSL, requires_core

# C5G7 UO2 data from the mg-mode-part-i notebook
UO2_TOTAL = [1.77949E-1, 3.29805E-1, 4.80388E-1, 5.54367E-1, 3.11801E-1,
             3.95168E-1, 5.64406E-1]
UO2_ABS = [8.0248E-03, 3.7174E-03, 2.6769E-02, 9.6236E-02, 3.0020E-02,
           1.1126E-01, 2.8278E-01]
UO2_SCATTER = [
    [1.27537E-1, 4.23780E-2, 9.43740E-6, 5.51630E-9, 0., 0., 0.],
    [0., 3.24456E-1, 1.63140E-3, 3.14270E-9, 0., 0., 0.],
    [0., 0., 4.50940E-1, 2.67920E-3, 0., 0., 0.],
    [0., 0., 0., 4.52565E-1, 5.56640E-3, 0., 0.],
    [0., 0., 0., 1.25250E-4, 2.71401E-1, 1.02550E-2, 1.00210E-8],
    [0., 0., 0., 0., 1.29680E-3, 2.65802E-1, 1.68090E-2],
    [0., 0., 0., 0., 0., 8.54580E-3, 2.73080E-1]]   # [g_in][g_out]


def uo2_xsdata():
    groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, 8))
    x = openmc.XSdata('uo2', groups)
    x.order = 0
    x.set_total(UO2_TOTAL, temperature=294.)
    x.set_absorption(UO2_ABS, temperature=294.)
    x.set_fission([7.21206E-03, 8.19301E-04, 6.45320E-03, 1.85648E-02,
                   1.78084E-02, 8.30348E-02, 2.16004E-01], temperature=294.)
    x.set_nu_fission([2.005998E-02, 2.027303E-03, 1.570599E-02, 4.518301E-02,
                      4.334208E-02, 2.020901E-01, 5.257105E-01], temperature=294.)
    x.set_chi([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0., 0., 0.],
              temperature=294.)
    scatter = np.rollaxis(np.array([UO2_SCATTER]), 0, 3)
    x.set_scatter_matrix(scatter, temperature=294.)
    return x


def test_energy_groups():
    g = openmc.mgxs.EnergyGroups('CASMO-2')
    assert g.num_groups == 2
    assert g.get_group(1.0) == 1 and g.get_group(0.1) == 2
    assert repr(g) == '<EnergyGroups: 2 groups (CASMO-2)>'
    assert openmc.mgxs.EnergyGroups([0., 0.625, 2e7]) == g
    assert 'MPACT-51' in openmc.mgxs.GROUP_STRUCTURES


def test_xsdata_validation():
    x = uo2_xsdata()
    assert x.fissionable
    with pytest.raises(ValueError):
        x.set_total([1.0, 2.0])
    with pytest.raises(ValueError):
        x.set_total(UO2_TOTAL, temperature=600.)
    lib = openmc.MGXSLibrary(x.energy_groups)
    lib.add_xsdata(x)
    other = openmc.XSdata('bad', openmc.mgxs.EnergyGroups('CASMO-2'))
    with pytest.raises(ValueError):
        lib.add_xsdata(other)
    assert lib.get_by_name('uo2') is x and lib.names == ['uo2']


def test_mpact_transport_correction():
    m = xs.mpact_xs(uo2_xsdata())
    assert np.allclose(m.total, UO2_TOTAL)
    # Out-scattering is unchanged, the in-group terms absorb the correction
    s = np.array(UO2_SCATTER).T         # [g_out][g_in]
    off = ~np.eye(7, dtype=bool)
    assert np.allclose(m.scatter[0][off], s[off])
    corr = np.array(UO2_TOTAL) - np.array(UO2_ABS) - s.sum(axis=0)
    assert np.allclose(np.diag(m.scatter[0]), np.diag(s) + corr)
    assert m.fissile and m.order == 0


def test_nu_fission_matrix_and_temperature():
    groups = openmc.mgxs.EnergyGroups('CASMO-2')
    x = openmc.XSdata('f', groups, temperatures=[300., 600.])
    x.order = 1
    for T in (300., 600.):
        x.set_total([0.5, 1.0], T)
        x.set_absorption([0.1, 0.4], T)
        x.set_nu_fission([[0.03, 0.01], [0.3, 0.1]], T)     # [g_in][g_out]
        scat = np.zeros((2, 2, 2))
        scat[:, :, 0] = [[0.3, 0.1], [0.0, T / 1000]]
        scat[:, :, 1] = [[0.01, 0.0], [0.0, 0.02]]
        x.set_scatter_matrix(scat, T)
    m = xs.mpact_xs(x, temperature=610.)
    assert np.allclose(m.nu_fission, [0.04, 0.4])
    assert np.allclose(m.chi, [0.75, 0.25])
    assert m.order == 1 and np.allclose(m.scatter[1], [[0.01, 0.0], [0.0, 0.02]])
    assert m.scatter[0][1, 1] == pytest.approx(1.0 - 0.4)       # 600 K data
    with pytest.warns(UserWarning):
        xs.mpact_xs(x, temperature=450.)


def test_xsl_roundtrip(tmp_path):
    lib = openmc.MGXSLibrary(uo2_xsdata().energy_groups)
    lib.add_xsdata(uo2_xsdata())
    path = tmp_path / 'uo2.xsl'
    xs.write_xsl(lib, path, names={'uo2': 'UO2'}, title='test')
    text = path.read_text().splitlines()
    assert text[0] == 'test' and text[1].split() == ['7', '1']
    read = xs.read_xsl(path)
    a, b = xs.mpact_xs(uo2_xsdata()), xs.mpact_xs(read.get_by_name('UO2'))
    for f in ('absorption', 'nu_fission', 'fission', 'chi', 'total'):
        assert np.allclose(getattr(a, f), getattr(b, f), rtol=1e-6)
    assert np.allclose(a.scatter[0], b.scatter[0], rtol=1e-6, atol=1e-12)
    # Group bounds: upper bounds, highest energy first, lowest lower bound 0 eV
    assert np.allclose(read.energy_groups.group_edges[1:],
                       lib.energy_groups.group_edges[1:], rtol=1e-6)
    assert read.energy_groups.group_edges[0] == 0.0


def test_read_reference_xsl(c5g7_library):
    lib = c5g7_library
    assert lib.names == ['UO2-3.3', 'MOX-4.3', 'MOX-7.0', 'MOX-8.7', 'FissCham',
                         'GuideTube', 'Moderator', 'CRod']
    uo2 = xs.mpact_xs(lib.get_by_name('UO2-3.3'))
    # The C5G7 benchmark transport cross sections equal absorption + scattering
    assert np.allclose(uo2.total, UO2_TOTAL, rtol=2e-5)
    assert np.allclose(uo2.absorption, UO2_ABS)
    assert uo2.scatter[0][1, 0] == pytest.approx(4.23780E-2)   # 1 -> 2
    assert np.allclose(lib.energy_groups.group_edges,
                       [0, 0.0635, 10., 1e2, 1e3, 5e5, 1e6, 2e7])


def test_hdf5_roundtrip(tmp_path, c5g7_library):
    pytest.importorskip('h5py')
    path = tmp_path / 'c5g7.h5'
    c5g7_library.export_to_hdf5(path)
    read = xs.load_library(path)
    assert sorted(read.names) == sorted(c5g7_library.names)   # HDF5 order is alphabetical
    for name in read.names:
        a = xs.mpact_xs(c5g7_library.get_by_name(name))
        b = xs.mpact_xs(read.get_by_name(name))
        assert np.allclose(a.total, b.total) and np.allclose(a.scatter[0], b.scatter[0])
    # load_library also detects .xsl files
    assert xs.load_library(C5G7_XSL).names == c5g7_library.names


@requires_core
def test_cpp_reader_matches(tmp_path, c5g7_library):
    import um2
    xs.write_xsl(c5g7_library, tmp_path / 'out.xsl')
    for path in (C5G7_XSL, tmp_path / 'out.xsl'):
        cpp = um2.UserXSLibrary(str(path))
        assert cpp.names == c5g7_library.names
        assert cpp.num_groups == 7
        for name in cpp.names:
            ref = xs.mpact_xs(c5g7_library.get_by_name(name))
            d = cpp.get_xsec(name)
            assert np.allclose(d['transport'], ref.total, rtol=1e-6)
            assert np.allclose(d['scatter_matrix'], ref.scatter[0], rtol=1e-6, atol=1e-14)
            assert np.allclose(d['chi'], ref.chi)
    with pytest.raises(um2.UM2Error):
        um2.UserXSLibrary(str(tmp_path / 'missing.xsl'))
