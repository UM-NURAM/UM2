# Python bindings for UM2 (the um2._um2 extension module) and the um2 package

# Prefer the Python that is first on PATH (e.g. of the active conda
# environment) over other installations.
set(Python_FIND_UNVERSIONED_NAMES FIRST)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "UM2 Python bindings: using ${Python_EXECUTABLE} (${Python_VERSION})")

# Let CMake find the pybind11 that is installed in the Python environment
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE UM2_PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if (UM2_PYBIND11_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${UM2_PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)

# UM2 and the gmsh Python module only share a gmsh model if they use the same
# libgmsh. Check that the gmsh module of this Python loads the libgmsh that
# UM2 links against.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import gmsh, os; print(os.path.realpath(gmsh.libpath))"
  OUTPUT_VARIABLE UM2_PYTHON_GMSH_LIB
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE UM2_PYTHON_GMSH_RESULT
  ERROR_QUIET)
get_filename_component(UM2_GMSH_LIB_REAL "${GMSH_LIB}" REALPATH)
if (NOT UM2_PYTHON_GMSH_RESULT EQUAL 0)
  message(WARNING "The gmsh Python module could not be imported by "
                  "${Python_EXECUTABLE}. Install gmsh in that Python environment.")
elseif (NOT UM2_PYTHON_GMSH_LIB STREQUAL UM2_GMSH_LIB_REAL)
  message(WARNING "The gmsh Python module of ${Python_EXECUTABLE} uses "
                  "${UM2_PYTHON_GMSH_LIB}, but UM2 links against ${UM2_GMSH_LIB_REAL}. "
                  "UM2 and the gmsh Python API will not share the same gmsh model. "
                  "Set GMSH_ROOT to the prefix of the Python environment "
                  "(e.g. GMSH_ROOT=$CONDA_PREFIX).")
endif()

pybind11_add_module(_um2 src/um2_python.cpp)
target_link_libraries(_um2 PRIVATE um2)
set_target_properties(_um2 PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})

if (SKBUILD)
  # pip install: put the extension and libum2 in the um2 package
  set_target_properties(_um2 PROPERTIES INSTALL_RPATH "$ORIGIN")
  install(TARGETS _um2 LIBRARY DESTINATION um2)
  install(TARGETS um2 LIBRARY DESTINATION um2)
  return()
endif()

#===============================================================================
# Build tree: <build>/python/um2 is a complete, importable um2 package
#===============================================================================
# The Python sources are copied (when they change) into the build directory,
# next to the extension, so that nothing is written to the source tree. Use the
# package with PYTHONPATH=<build>/python.

set(UM2_PYTHON_DIR "${PROJECT_BINARY_DIR}/python")
set_target_properties(_um2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${UM2_PYTHON_DIR}/um2")

file(GLOB_RECURSE UM2_PYTHON_SOURCES CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/um2/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/um2/*.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/um2/LICENSE*")
list(REMOVE_DUPLICATES UM2_PYTHON_SOURCES)
set(UM2_PYTHON_OUTPUTS "")
foreach (file IN LISTS UM2_PYTHON_SOURCES)
  set(output "${UM2_PYTHON_DIR}/${file}")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${output}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
    COMMENT "Copying python/${file}"
    VERBATIM)
  list(APPEND UM2_PYTHON_OUTPUTS "${output}")
endforeach()
add_custom_target(um2_python_package ALL DEPENDS ${UM2_PYTHON_OUTPUTS})
add_dependencies(_um2 um2_python_package)

# Environment script: source <build>/python/env.sh
file(WRITE "${UM2_PYTHON_DIR}/env.sh"
  "# Source this file to use the um2 Python package of this build\n"
  "export PYTHONPATH=\"${UM2_PYTHON_DIR}\${PYTHONPATH:+:\$PYTHONPATH}\"\n"
  "# Do not write bytecode (__pycache__) into the source tree when running the\n"
  "# tests in <UM2>/python/tests\n"
  "export PYTHONDONTWRITEBYTECODE=1\n")

# The Python tests (run with ctest, or see python/README.md). Python and pytest
# write no bytecode or cache files into the source tree.
if (UM2_BUILD_TESTS)
  add_test(NAME python_tests
    COMMAND "${Python_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            -m "not slow" -p no:cacheprovider -q
    WORKING_DIRECTORY "${UM2_PYTHON_DIR}")
  set_tests_properties(python_tests PROPERTIES
    ENVIRONMENT "PYTHONPATH=${UM2_PYTHON_DIR};PYTHONDONTWRITEBYTECODE=1;MPLBACKEND=Agg"
    TIMEOUT 1800)
endif()
