"""Model container (compatible with :class:`openmc.Model`)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ...errors import UM2NotSupportedError
from .. import _stubs
from ..checkvalue import check_type
from ..geometry import Geometry
from ..material import Material, Materials


class Model:
    """Model container.

    UM2 uses the :attr:`geometry` and :attr:`materials` of a model. The
    settings, tallies and plots are accepted for compatibility with OpenMC but
    ignored.

    Parameters
    ----------
    geometry : openmc.Geometry, optional
        Geometry information
    materials : openmc.Materials, optional
        Materials information
    settings : openmc.Settings, optional
        Settings information (ignored by UM2)
    tallies : openmc.Tallies, optional
        Tallies information (ignored by UM2)
    plots : openmc.Plots, optional
        Plot information (ignored by UM2)
    description : str, optional
        Description of the model
    """

    def __init__(self, geometry=None, materials=None, settings=None, tallies=None,
                 plots=None, description=''):
        self.geometry = Geometry() if geometry is None else geometry
        self.materials = Materials() if materials is None else materials
        self._settings = settings
        self._tallies = tallies
        self._plots = plots
        self.description = description

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        check_type('geometry', geometry, Geometry)
        self._geometry = geometry

    @property
    def materials(self) -> Materials:
        return self._materials

    @materials.setter
    def materials(self, materials):
        check_type('materials', materials, Iterable, Material)
        if isinstance(materials, Materials):
            self._materials = materials
        else:
            if not hasattr(self, '_materials'):
                self._materials = Materials()
            del self._materials[:]
            for mat in materials:
                self._materials.append(mat)

    @property
    def settings(self):
        if self._settings is None:
            self._settings = _stubs.Settings()
        return self._settings

    @settings.setter
    def settings(self, settings):
        self._settings = settings

    @property
    def tallies(self):
        if self._tallies is None:
            self._tallies = _stubs.Tallies()
        return self._tallies

    @tallies.setter
    def tallies(self, tallies):
        self._tallies = tallies

    @property
    def plots(self):
        if self._plots is None:
            self._plots = _stubs.Plots()
        return self._plots

    @plots.setter
    def plots(self, plots):
        self._plots = plots

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, description):
        check_type('description', description, str)
        self._description = description

    @property
    def bounding_box(self):
        return self.geometry.bounding_box

    def get_all_materials(self) -> Materials:
        """Return the materials of the model, or, if none were given, all of
        the materials in the geometry. The cross sections path is preserved."""
        if len(self.materials) > 0:
            return self.materials
        materials = Materials(self.geometry.get_all_materials().values())
        materials.cross_sections = self.materials.cross_sections
        return materials

    def export_to_xml(self, directory='.', remove_surfs=False, nuclides_to_ignore=None):
        """Export the materials and geometry to materials.xml and geometry.xml.

        Settings, tallies and plots are ignored by UM2 and are not exported.
        """
        d = Path(directory)
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
        self.get_all_materials().export_to_xml(d, nuclides_to_ignore=nuclides_to_ignore)
        self.geometry.export_to_xml(d, remove_surfs=remove_surfs)

    def run(self, *args, **kwargs):
        """Ignored by UM2 (openmc.Model.run)."""
        _stubs._ignored('openmc.Model.run')

    def plot(self, *args, **kwargs):
        """Ignored by UM2 (openmc.Model.plot)."""
        _stubs._ignored('openmc.Model.plot')

    def plot_geometry(self, *args, **kwargs):
        """Ignored by UM2 (openmc.Model.plot_geometry)."""
        _stubs._ignored('openmc.Model.plot_geometry')

    def calculate_volumes(self, *args, **kwargs):
        """Ignored by UM2 (openmc.Model.calculate_volumes)."""
        _stubs._ignored('openmc.Model.calculate_volumes')

    def deplete(self, *args, **kwargs):
        """Not supported by UM2."""
        raise UM2NotSupportedError('Model.deplete requires running OpenMC.')

    def convert_to_multigroup(self, *args, **kwargs):
        """Not supported by UM2."""
        raise UM2NotSupportedError('Model.convert_to_multigroup requires running '
                                   'OpenMC.')
