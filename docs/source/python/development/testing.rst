Running the Tests
=================

Install the optional test dependencies:

.. code-block:: bash

   python -m pip install -e './python[test]' --no-build-isolation

Run the Python test suite from the repository root:

.. code-block:: bash

   python -m pytest python/tests -q --tb=short

Some tests require the compiled C++ bindings, optional resources,
or external OpenMC data.

The OpenMC-dependent integration tests can be enabled with:

.. code-block:: bash

   UM2_RUN_OPENMC_TESTS=1 \
   python -m pytest python/tests/test_full_pipeline.py -v

These tests also require the ``OPENMC_CROSS_SECTIONS`` environment
variable to point to a valid OpenMC continuous-energy cross-section
library. Without it, tests requiring that data are skipped.
