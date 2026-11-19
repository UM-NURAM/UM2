Building the Documentation
==========================

The documentation is built using Sphinx.

Install the documentation dependencies:

.. code-block:: bash

   python -m pip install -r docs/requirements-rtd.txt

Build the HTML documentation:

.. code-block:: bash

   python -m sphinx -b html docs/source docs/_build/html

The generated website is available at:

.. code-block:: text

   docs/_build/html/index.html

To preview it locally:

.. code-block:: bash

   python -m http.server 8000 --directory docs/_build/html

Then open ``http://localhost:8000`` in a browser.
