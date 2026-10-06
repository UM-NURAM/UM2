"""Regression test: the C++ bindings must not hold the GIL while UM2/gmsh print a lot of output.

A Jupyter kernel redirects the C-level stdout into a pipe that a *Python thread* drains. If a
binding holds the GIL during a long call that prints more than the pipe can hold (64 KB), the
drain thread cannot run, the write blocks, and the kernel hangs forever. This test recreates
that situation in a child process (so a failure cannot hang the test run).
"""

import subprocess
import sys

import pytest

from conftest import requires_core

pytestmark = requires_core

CHILD = r'''
import os, threading, time

saved = os.dup(1)                 # the real stdout, for reporting the result
r, w = os.pipe()
os.dup2(w, 1)                     # C-level stdout is now a pipe ...
os.close(w)
drained = [0]

def drain():                      # ... that is drained by a Python thread, as in a Jupyter kernel
    while True:
        data = os.read(r, 1000)
        if not data:
            return
        drained[0] += len(data)

threading.Thread(target=drain, daemon=True).start()

import gmsh
import um2

um2.initialize()                  # gmsh prints its progress to stdout (General.Terminal = 1)
core = um2._core()
# 2025 small disjoint squares. The gmsh Python API (ctypes) releases the GIL, so this is fast and
# harmless no matter how much it prints.
for i in range(45):
    for j in range(45):
        gmsh.model.occ.addRectangle(float(i), float(j), 0.0, 0.9, 0.9)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], name='Material_x')
core.gmsh.set_global_mesh_size(0.5)
time.sleep(1.0)
before = drained[0]
# One call into the UM2 bindings prints a line per surface (and more): well over a pipe buffer.
core.gmsh.generate_mesh(core.MeshType.Tri, 0)
time.sleep(1.0)
os.write(saved, f'DONE drained_during_call={drained[0] - before}\n'.encode())
'''


def test_no_deadlock_when_stdout_is_drained_by_a_python_thread():
    try:
        result = subprocess.run([sys.executable, '-c', CHILD], capture_output=True,
                                text=True, timeout=90)
    except subprocess.TimeoutExpired:
        pytest.fail('generate_mesh hung: a binding holds the GIL while writing to a stdout '
                    'pipe that a Python thread must drain (the Jupyter kernel deadlock)')
    assert result.returncode == 0, result.stderr[-2000:]
    done = [line for line in result.stdout.splitlines() if line.startswith('DONE')]
    assert done, result.stdout[-2000:]
    drained = int(done[0].split('=')[1])
    # Make sure the test is not vacuous: much more than one pipe buffer (64 KB) of output
    assert drained > 100_000, f'only {drained} bytes were written; the test is too small'
