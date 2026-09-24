"""Material colors.

UM2 colors gmsh entities by material, which makes the model easy to inspect
in the gmsh GUI and is used by UM2 when it creates materials from gmsh physical
groups. Colors are assigned deterministically from the material name unless
they are set explicitly with :func:`set_material_color`.
"""

import colorsys
import hashlib

_COLORS = {}

#: Colors used for common material names (matched case-insensitively)
_DEFAULTS = {
    'void': (255, 255, 255),
}


def set_material_color(name, color):
    """Set the color of a material.

    Parameters
    ----------
    name : str or um2.openmc.Material
        Material, or the UM2 name of a material
    color : tuple of int
        (r, g, b) color with components in [0, 255]
    """
    if not isinstance(name, str):
        name = name.name
    r, g, b = (int(c) for c in color)
    for c in (r, g, b):
        if not 0 <= c <= 255:
            raise ValueError(f'Color components must be in [0, 255], got {color}')
    _COLORS[name] = (r, g, b)


def material_color(name):
    """Return the (r, g, b) color of a material."""
    if name in _COLORS:
        return _COLORS[name]
    if name.lower() in _DEFAULTS:
        return _DEFAULTS[name.lower()]
    # Hash the name to a hue; use fixed saturation and value so colors are
    # distinguishable.
    h = int.from_bytes(hashlib.md5(name.encode()).digest()[:4], 'little') / 2**32
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.9)
    return (int(255*r), int(255*g), int(255*b))
