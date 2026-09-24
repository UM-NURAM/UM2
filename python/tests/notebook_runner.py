"""Run the code cells of OpenMC example notebooks with um2.openmc as openmc.

The notebooks come from https://github.com/openmc-dev/openmc-notebooks. Set the
OPENMC_NOTEBOOKS environment variable to the path of a checkout (by default,
a checkout next to the UM2 repository is used).
"""

import json
import os
from pathlib import Path
import sys
import warnings

REPO = Path(__file__).resolve().parents[2]


def notebooks_dir():
    path = os.environ.get('OPENMC_NOTEBOOKS')
    if path:
        return Path(path)
    return REPO.parent / 'openmc-notebooks'


def code_cells(notebook):
    """Return the code cells of a notebook, as a list of (index, source)."""
    nb = json.loads(Path(notebook).read_text())
    cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                source = ''.join(source)
            cells.append((i, source))
    return cells


def _strip_ipython(source):
    """Remove IPython magics and shell commands."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(('%', '!')):
            indent = line[:len(line) - len(stripped)]
            lines.append(indent + 'pass')
        else:
            lines.append(line)
    return '\n'.join(lines)


def _reset_openmc_modules():
    for name in list(sys.modules):
        if name == 'openmc' or name.startswith('openmc.'):
            del sys.modules[name]


def run_notebook(name, stop=None, skip=(), replace=None, workdir=None):
    """Execute the code cells of a notebook with ``openmc`` = um2.openmc.

    Parameters
    ----------
    name : str
        Notebook name (without .ipynb)
    stop : int, optional
        Execute cells with notebook cell index < stop
    skip : iterable of int
        Notebook cell indices to skip (e.g. cells that need OpenMC results)
    replace : dict, optional
        Mapping of cell index to a list of (old, new) source replacements
    workdir : path-like, optional
        Directory to run in (files written by the notebook go here). The
        notebook directory is used to resolve data files such as c5g7.h5.

    Returns
    -------
    dict
        The namespace after executing the cells
    """
    import um2.openmc
    import um2.openmc.mixin

    path = notebooks_dir() / f'{name}.ipynb'
    _reset_openmc_modules()
    um2.openmc.as_openmc()
    um2.openmc.reset_auto_ids()
    replace = replace or {}
    namespace = {'__name__': '__notebook__'}
    cwd = os.getcwd()
    os.chdir(workdir or path.parent)
    try:
        for index, source in code_cells(path):
            if stop is not None and index >= stop:
                break
            if index in skip:
                continue
            for old, new in replace.get(index, ()):
                if old not in source:
                    raise ValueError(f'{name} cell {index}: "{old}" not found')
                source = source.replace(old, new)
            source = _strip_ipython(source)
            code = compile(source, f'{name}.ipynb[{index}]', 'exec')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exec(code, namespace)
    finally:
        os.chdir(cwd)
        _reset_openmc_modules()
    return namespace
