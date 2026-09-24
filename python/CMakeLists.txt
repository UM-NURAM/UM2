# Python bindings for UM2 (the um2._um2 extension module)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
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

pybind11_add_module(_um2 src/um2_python.cpp)
target_link_libraries(_um2 PRIVATE um2)
set_target_properties(_um2 PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})

if (SKBUILD)
  # pip install: put the extension and libum2 in the um2 package
  set_target_properties(_um2 PROPERTIES INSTALL_RPATH "$ORIGIN")
  install(TARGETS _um2 LIBRARY DESTINATION um2)
  install(TARGETS um2 LIBRARY DESTINATION um2)
else()
  # Development build: put the extension next to the Python sources, so that
  # the package can be used with PYTHONPATH=<UM2>/python
  set_target_properties(_um2 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/um2")
endif()
