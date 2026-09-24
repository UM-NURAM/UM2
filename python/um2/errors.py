"""Exceptions raised by UM2."""


class UM2Error(RuntimeError):
    """An error reported by the UM2 C++ library."""


class UM2NotSupportedError(NotImplementedError):
    """A feature of the OpenMC API that UM2 does not support.

    UM2 generates 2D (Z-invariant) meshes, so inherently 3D features (e.g.
    spheres, cones and axial layers) are not supported.
    """
