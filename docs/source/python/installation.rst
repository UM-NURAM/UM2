============
Installation
============

The UM2 Python API provides tools for constructing OpenMC-compatible
geometries, generating unstructured meshes, handling multigroup cross
sections, and exporting MPACT models.

The Python bindings are compiled from the UM2 C++ library. Building
the package therefore requires both Python dependencies and the
native dependencies of UM2.

Requirements
============

The Python package requires:

* Python 3.10 or newer
* A C++ compiler supporting the UM2 source code
* CMake
* Gmsh and its shared library
* HDF5 with C++ support
* Additional native dependencies required by the UM2 CMake build

The Python package uses ``scikit-build-core`` and ``pybind11`` for
compilation. Its runtime dependencies include NumPy, Gmsh, and h5py.

The native dependencies must be installed and discoverable by CMake
before the Python extension can be built.

Creating a Python Environment
=============================

From the root of the UM2 repository, create and activate a virtual
environment:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

Upgrade pip and install the Python build dependencies:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install scikit-build-core pybind11 numpy gmsh h5py

The Gmsh Python module and the UM2 C++ extension must use compatible
versions of the Gmsh shared library.

Building and Installing UM2
===========================

The Python package is located in the ``python/`` directory.

For an editable developer installation, use:

.. code-block:: bash

   GMSH_ROOT="$VIRTUAL_ENV" \
   CMAKE_PREFIX_PATH="$VIRTUAL_ENV" \
   python -m pip install --no-build-isolation -e ./python

This invokes the UM2 CMake build and installs the Python package in
editable mode.

The package configuration enables the Python bindings and disables
the optional MPACT cross-section-library dependency.

An editable installation allows changes to Python source files to
be used without reinstalling the package. Changes to C++ bindings
still require rebuilding the extension.

Verifying the Installation
==========================

Verify that the Python package is importable:

.. code-block:: bash

   python -c "import um2; print(um2.__file__)"

Then verify the compiled extension:

.. code-block:: bash

   python -c "import um2._um2; print('UM2 C++ bindings loaded')"

Both commands should complete without errors.

Gmsh Shared Library Configuration
==================================

On some Linux systems, the UM2 extension may be compiled successfully
but fail to load because the dynamic linker cannot locate the Gmsh
shared library.

For example:

.. code-block:: text

   ImportError: libgmsh.so.4.15: cannot open shared object file

First, locate the library:

.. code-block:: bash

   find "$VIRTUAL_ENV" -name 'libgmsh.so*' -print

If the required shared library is located in ``$VIRTUAL_ENV/lib``,
add that directory to the runtime library search path:

.. code-block:: bash

   export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

Then retry:

.. code-block:: bash

   python -c "import um2._um2; print('UM2 C++ bindings loaded')"

This environment variable applies to the current shell session.
It may need to be configured again when opening another terminal.

The required Gmsh library version depends on the version used to
build the extension. Rebuilding against a different version may
require updating the runtime library configuration.
