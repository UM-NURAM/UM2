// Python bindings for UM2.
//
// These bindings expose the parts of UM2 that are needed to mesh and export a
// gmsh model from Python: materials with macroscopic cross sections, the MPACT
// spatial partition (mpact::Model), coarse grid overlay, mesh sizing and
// generation, and I/O. The gmsh model itself is shared with the gmsh Python
// API, since both use the same libgmsh.
//
// The GIL is released during every call into UM2/gmsh C++ (see the
// py::gil_scoped_release in each function below). UM2 and gmsh print a lot to the C-level
// stdout, which Jupyter kernels redirect into a pipe that is drained by a Python thread. If
// the GIL were held during the call, that thread could not run, the 64 KB pipe would fill
// up, the write would block, and the kernel would hang (a deadlock). The functions below
// only use C++ objects after the GIL is released.
//
// UM2 reports errors through its logger. The bindings disable exit-on-error,
// clear the recorded errors before each call, and raise um2.UM2Error if an
// error was logged during the call.

#include <um2.hpp>
#include <um2/common/logger.hpp>
#include <um2/gmsh/io.hpp>
#include <um2/gmsh/mesh.hpp>
#include <um2/gmsh/model.hpp>
#include <um2/mpact/model.hpp>
#include <um2/physics/material.hpp>
#include <um2/physics/mpact_user_xs.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace
{

//==============================================================================
// Error handling
//==============================================================================

class UM2Exception : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

// Clears the logger's error record on construction. check() throws if an
// error was logged since then.
class ErrorScope
{
public:
  ErrorScope() noexcept
  {
    um2::logger::exit_on_error = false;
    um2::logger::clearErrors();
  }

  static void
  check()
  {
    if (um2::logger::error_count > 0) {
      std::string msg(um2::logger::first_error);
      if (um2::logger::error_count > 1) {
        msg += " (and " + std::to_string(um2::logger::error_count - 1) +
               " other error(s); see the log for details)";
      }
      um2::logger::clearErrors();
      throw UM2Exception(msg);
    }
  }
};

//==============================================================================
// Conversions
//==============================================================================

auto
toString(um2::String const & s) -> std::string
{
  return {s.data(), static_cast<size_t>(s.size())};
}

auto
toUM2String(std::string const & s) -> um2::String
{
  return um2::String(s.c_str());
}

template <class T>
auto
toVector(std::vector<T> const & v) -> um2::Vector<T>
{
  um2::Vector<T> result(static_cast<Int>(v.size()));
  for (size_t i = 0; i < v.size(); ++i) {
    result[static_cast<Int>(i)] = v[i];
  }
  return result;
}

auto
toVector2D(std::vector<std::vector<Int>> const & v) -> um2::Vector<um2::Vector<Int>>
{
  um2::Vector<um2::Vector<Int>> result(static_cast<Int>(v.size()));
  for (size_t i = 0; i < v.size(); ++i) {
    result[static_cast<Int>(i)] = toVector(v[i]);
  }
  return result;
}

template <class T>
auto
toNumpy(um2::Vector<T> const & v) -> py::array_t<T>
{
  py::array_t<T> result(static_cast<py::ssize_t>(v.size()));
  auto r = result.template mutable_unchecked<1>();
  for (Int i = 0; i < v.size(); ++i) {
    r(i) = v[i];
  }
  return result;
}

using FloatArray = py::array_t<Float, py::array::c_style | py::array::forcecast>;

auto
toVector(FloatArray const & a, py::ssize_t n, char const * name) -> um2::Vector<Float>
{
  if (a.ndim() != 1 || a.shape(0) != n) {
    throw std::invalid_argument(std::string(name) + " must have shape (" +
                                std::to_string(n) + ",)");
  }
  auto r = a.unchecked<1>();
  um2::Vector<Float> result(static_cast<Int>(n));
  for (py::ssize_t i = 0; i < n; ++i) {
    result[static_cast<Int>(i)] = r(i);
  }
  return result;
}

//==============================================================================
// Cross sections
//==============================================================================

// Set the macroscopic cross section of a material.
//  ss[i, j] is the cross section for scattering from group j to group i.
//  The total scattering cross section is computed from the columns of ss.
void
setXSec(um2::Material & mat, FloatArray const & a, FloatArray const & f,
        FloatArray const & nuf, FloatArray const & tr, FloatArray const & ss)
{
  if (a.ndim() != 1) {
    throw std::invalid_argument("absorption must be a 1D array");
  }
  py::ssize_t const n = a.shape(0);
  if (n <= 0) {
    throw std::invalid_argument("cross sections must have at least one group");
  }
  if (ss.ndim() != 2 || ss.shape(0) != n || ss.shape(1) != n) {
    throw std::invalid_argument("scatter matrix must have shape (G, G)");
  }
  auto const num_groups = static_cast<Int>(n);
  um2::XSec xs(num_groups);
  xs.isMacro() = true;
  xs.a() = toVector(a, n, "absorption");
  xs.f() = toVector(f, n, "fission");
  xs.nuf() = toVector(nuf, n, "nu-fission");
  xs.tr() = toVector(tr, n, "transport");
  auto s = ss.unchecked<2>();
  for (Int i = 0; i < num_groups; ++i) {
    for (Int j = 0; j < num_groups; ++j) {
      xs.ss()(i, j) = s(i, j);
    }
  }
  for (Int j = 0; j < num_groups; ++j) {
    Float total = 0;
    for (Int i = 0; i < num_groups; ++i) {
      total += xs.ss()(i, j);
    }
    xs.s()[j] = total;
  }
  for (Int g = 0; g < num_groups; ++g) {
    if (xs.f()[g] > 0 || xs.nuf()[g] > 0) {
      xs.isFissile() = true;
    }
  }
  py::gil_scoped_release nogil;
  ErrorScope scope;
  xs.validate();
  mat.xsec() = um2::move(xs);
  ErrorScope::check();
}

auto
scatterMatrix(um2::XSec const & xs) -> py::array_t<Float>
{
  Int const n = xs.numGroups();
  py::array_t<Float> result({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
  auto r = result.mutable_unchecked<2>();
  for (Int i = 0; i < n; ++i) {
    for (Int j = 0; j < n; ++j) {
      r(i, j) = xs.ss()(i, j);
    }
  }
  return result;
}

auto
xsecDict(um2::XSec const & xs) -> py::dict
{
  py::dict d;
  d["absorption"] = toNumpy(xs.a());
  d["fission"] = toNumpy(xs.f());
  d["nu_fission"] = toNumpy(xs.nuf());
  d["transport"] = toNumpy(xs.tr());
  d["scatter"] = toNumpy(xs.s());
  d["scatter_matrix"] = scatterMatrix(xs);
  d["fissile"] = xs.isFissile();
  return d;
}

auto
meshTypeName(um2::MeshType t) -> char const *
{
  switch (t) {
  case um2::MeshType::Tri:
    return "Tri";
  case um2::MeshType::Quad:
    return "Quad";
  case um2::MeshType::TriQuad:
    return "TriQuad";
  case um2::MeshType::QuadraticTri:
    return "QuadraticTri";
  case um2::MeshType::QuadraticQuad:
    return "QuadraticQuad";
  case um2::MeshType::QuadraticTriQuad:
    return "QuadraticTriQuad";
  default:
    return "Invalid";
  }
}

} // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PYBIND11_MODULE(_um2, m)
{
  m.doc() = "UM2 C++ bindings. Use the um2 package rather than this module directly.";

  // Translate C++ errors to um2.errors.UM2Error
  py::object um2_error = py::module_::import("um2.errors").attr("UM2Error");
  static py::handle error_type = um2_error.release();
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (UM2Exception const & e) {
      PyErr_SetString(error_type.ptr(), e.what());
    }
  });

  //============================================================================
  // Library
  //============================================================================

  m.def(
      "initialize",
      []() {
        um2::initialize();
        um2::logger::exit_on_error = false;
      },
      "Initialize UM2 (and gmsh, if it is not already initialized).");

  m.def(
      "finalize",
      [](bool finalize_gmsh) {
        um2::logger::info("Finalizing UM2");
        if (finalize_gmsh) {
          um2::finalize();
        }
      },
      py::arg("finalize_gmsh") = false,
      "Finalize UM2. gmsh is only finalized if finalize_gmsh is True.");

  m.def(
      "set_log_level", [](int32_t level) { um2::logger::level = level; }, py::arg("level"),
      "Set the verbosity of the UM2 log: 0 = off, 1 = error, 2 = warn, 3 = info, "
      "4 = debug.");
  m.def("get_log_level", []() { return um2::logger::level; });
  m.def(
      "set_log_timestamped", [](bool value) { um2::logger::timestamped = value; },
      py::arg("value"));
  m.def(
      "set_log_colorized", [](bool value) { um2::logger::colorized = value; },
      py::arg("value"));

  m.attr("float_size") = sizeof(Float);
  m.attr("int_size") = sizeof(Int);
  m.attr("max_materials") = 127;

  //============================================================================
  // Mesh types
  //============================================================================

  py::enum_<um2::MeshType>(m, "MeshType", "Type of mesh to generate.")
      .value("Tri", um2::MeshType::Tri)
      .value("Quad", um2::MeshType::Quad)
      .value("TriQuad", um2::MeshType::TriQuad)
      .value("QuadraticTri", um2::MeshType::QuadraticTri)
      .value("QuadraticQuad", um2::MeshType::QuadraticQuad)
      .value("QuadraticTriQuad", um2::MeshType::QuadraticTriQuad);

  //============================================================================
  // Material
  //============================================================================

  py::class_<um2::Material>(m, "Material",
                            "A UM2 material with a macroscopic cross section.")
      .def(py::init<>())
      .def(py::init([](std::string const & name) {
             um2::Material mat;
             mat.setName(toUM2String(name));
             return mat;
           }),
           py::arg("name"))
      .def_property(
          "name", [](um2::Material const & mat) { return toString(mat.getName()); },
          [](um2::Material & mat, std::string const & name) {
            mat.setName(toUM2String(name));
          })
      .def_property(
          "color",
          [](um2::Material const & mat) {
            auto const c = mat.getColor();
            return std::make_tuple(c.r(), c.g(), c.b(), c.a());
          },
          [](um2::Material & mat, py::tuple const & c) {
            if (c.size() != 3 && c.size() != 4) {
              throw std::invalid_argument("color must be (r, g, b) or (r, g, b, a)");
            }
            int const alpha = c.size() == 4 ? c[3].cast<int>() : 255;
            mat.setColor(um2::Color(c[0].cast<int>(), c[1].cast<int>(),
                                    c[2].cast<int>(), alpha));
          })
      .def_property("temperature", &um2::Material::getTemperature,
                    &um2::Material::setTemperature)
      .def_property("density", &um2::Material::getDensity, &um2::Material::setDensity)
      .def("has_xsec", &um2::Material::hasXSec)
      .def_property_readonly("num_groups",
                             [](um2::Material const & mat) { return mat.xsec().numGroups(); })
      .def("set_xsec", &setXSec, py::arg("absorption"), py::arg("fission"),
           py::arg("nu_fission"), py::arg("transport"), py::arg("scatter_matrix"),
           "Set the macroscopic cross section. scatter_matrix[i, j] is the cross "
           "section for scattering from group j to group i. The total cross "
           "section used by UM2 is absorption + sum of the scattering matrix "
           "column.")
      .def("get_xsec", [](um2::Material const & mat) { return xsecDict(mat.xsec()); },
           "Return the macroscopic cross section as a dict of numpy arrays.")
      .def("validate_xsec",
           [](um2::Material const & mat) {
             py::gil_scoped_release nogil;
             ErrorScope scope;
             mat.validateXSec();
             ErrorScope::check();
           })
      .def("__repr__", [](um2::Material const & mat) {
        return "<um2.Material '" + toString(mat.getName()) + "'>";
      });

  //============================================================================
  // MPACT user cross section library (.xsl)
  //============================================================================

  py::class_<um2::mpact::UserXSLibrary>(m, "UserXSLibrary",
                                        "An MPACT user (macroscopic) cross section "
                                        "library (.xsl).")
      .def(py::init([](std::string const & filename) {
             py::gil_scoped_release nogil;
             ErrorScope scope;
             um2::mpact::UserXSLibrary lib(toUM2String(filename));
             ErrorScope::check();
             return lib;
           }),
           py::arg("filename"))
      .def_property_readonly("title",
                             [](um2::mpact::UserXSLibrary const & lib) {
                               return toString(lib.title());
                             })
      .def_property_readonly("group_bounds",
                             [](um2::mpact::UserXSLibrary const & lib) {
                               return toNumpy(lib.groupBounds());
                             })
      .def_property_readonly("num_groups", &um2::mpact::UserXSLibrary::numGroups)
      .def_property_readonly("names",
                             [](um2::mpact::UserXSLibrary const & lib) {
                               std::vector<std::string> names;
                               for (auto const & name : lib.names()) {
                                 names.push_back(toString(name));
                               }
                               return names;
                             })
      .def_property_readonly("scattering_orders",
                             [](um2::mpact::UserXSLibrary const & lib) {
                               return toNumpy(lib.scatteringOrders());
                             })
      .def(
          "get_xsec",
          [](um2::mpact::UserXSLibrary const & lib, std::string const & name) {
            Int const i = lib.getIndex(toUM2String(name));
            if (i < 0) {
              throw py::key_error("Cross section set not found: " + name);
            }
            py::dict d = xsecDict(lib.xsecs()[i]);
            d["chi"] = toNumpy(lib.chis()[i]);
            return d;
          },
          py::arg("name"));

  //============================================================================
  // MPACT model
  //============================================================================

  py::module_ mpact = m.def_submodule("mpact", "MPACT spatial partition");

  py::class_<um2::mpact::Model>(mpact, "Model",
                                "The MPACT spatial partition: core, assemblies, "
                                "lattices, ray tracing modules and coarse cells.")
      .def(py::init<>())
      .def(
          "add_material",
          [](um2::mpact::Model & model, um2::Material const & mat, bool validate) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            Int const id = model.addMaterial(mat, validate);
            ErrorScope::check();
            return id;
          },
          py::arg("material"), py::arg("validate") = true)
      .def(
          "add_coarse_cell",
          [](um2::mpact::Model & model, std::tuple<Float, Float> xy_extents) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            Int const id = model.addCoarseCell(
                um2::Vec2F(std::get<0>(xy_extents), std::get<1>(xy_extents)));
            ErrorScope::check();
            return id;
          },
          py::arg("xy_extents"))
      .def(
          "add_rtm",
          [](um2::mpact::Model & model, std::vector<std::vector<Int>> const & ids) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            Int const id = model.addRTM(toVector2D(ids));
            ErrorScope::check();
            return id;
          },
          py::arg("coarse_cell_ids"),
          "Add a ray tracing module. Rows are ordered from top (largest y) to "
          "bottom.")
      .def(
          "add_lattice",
          [](um2::mpact::Model & model, std::vector<std::vector<Int>> const & ids) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            Int const id = model.addLattice(toVector2D(ids));
            ErrorScope::check();
            return id;
          },
          py::arg("rtm_ids"),
          "Add a lattice of RTMs. Rows are ordered from top (largest y) to bottom.")
      .def(
          "add_assembly",
          [](um2::mpact::Model & model, std::vector<Int> const & ids,
             std::vector<Float> const & z) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            Int const id = model.addAssembly(toVector(ids), toVector(z));
            ErrorScope::check();
            return id;
          },
          py::arg("lattice_ids"), py::arg("z") = std::vector<Float>{-0.5, 0.5})
      .def(
          "add_core",
          [](um2::mpact::Model & model, std::vector<std::vector<Int>> const & ids) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            Int const id = model.addCore(toVector2D(ids));
            ErrorScope::check();
            return id;
          },
          py::arg("assembly_ids"),
          "Add the core. Rows are ordered from top (largest y) to bottom.")
      .def(
          "add_coarse_grid",
          [](um2::mpact::Model & model, std::tuple<Float, Float> xy_extents,
             std::tuple<Int, Int> num_cells) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            model.addCoarseGrid(
                um2::Vec2F(std::get<0>(xy_extents), std::get<1>(xy_extents)),
                um2::Vec2I(std::get<0>(num_cells), std::get<1>(num_cells)));
            ErrorScope::check();
          },
          py::arg("xy_extents"), py::arg("num_cells"))
      .def(
          "import_coarse_cell_meshes",
          [](um2::mpact::Model & model, std::string const & filename) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            model.importCoarseCellMeshes(toUM2String(filename));
            ErrorScope::check();
          },
          py::arg("filename"))
      .def(
          "write",
          [](um2::mpact::Model const & model, std::string const & filename,
             bool write_knudsen_data, bool write_xsec_data) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            model.write(toUM2String(filename), write_knudsen_data, write_xsec_data);
            ErrorScope::check();
          },
          py::arg("filename"), py::arg("write_knudsen_data") = false,
          py::arg("write_xsec_data") = false,
          "Write the model to an XDMF file (plus the MPACT input, material volume "
          "and cross section files).")
      .def(
          "read",
          [](um2::mpact::Model & model, std::string const & filename) {
            py::gil_scoped_release nogil;
            ErrorScope scope;
            model.read(toUM2String(filename));
            ErrorScope::check();
          },
          py::arg("filename"))
      .def_property_readonly("num_coarse_cells", &um2::mpact::Model::numCoarseCells)
      .def_property_readonly("num_rtms", &um2::mpact::Model::numRTMs)
      .def_property_readonly("num_lattices", &um2::mpact::Model::numLattices)
      .def_property_readonly("num_assemblies", &um2::mpact::Model::numAssemblies)
      .def_property_readonly("num_coarse_cells_total",
                             &um2::mpact::Model::numCoarseCellsTotal)
      .def_property_readonly("num_fine_cells_total",
                             &um2::mpact::Model::numFineCellsTotal)
      .def_property_readonly("materials",
                             [](um2::mpact::Model const & model) {
                               std::vector<um2::Material> mats;
                               for (auto const & mat : model.materials()) {
                                 mats.push_back(mat);
                               }
                               return mats;
                             })
      .def(
          "core_children",
          [](um2::mpact::Model const & model) {
            // Assembly IDs, rows ordered from bottom (y index 0) to top
            auto const & core = model.getCore();
            std::vector<std::vector<Int>> ids;
            if (core.children().empty()) {
              return ids;
            }
            for (Int iy = 0; iy < core.grid().numCells(1); ++iy) {
              std::vector<Int> row;
              for (Int ix = 0; ix < core.grid().numCells(0); ++ix) {
                row.push_back(core.getChild(ix, iy));
              }
              ids.push_back(row);
            }
            return ids;
          },
          "Assembly IDs of the core, indexed [iy][ix] with iy = 0 at the bottom.")
      .def(
          "coarse_cell",
          [](um2::mpact::Model const & model, Int i) {
            if (i < 0 || i >= model.numCoarseCells()) {
              throw py::index_error("coarse cell index out of range");
            }
            auto const & cc = model.getCoarseCell(i);
            py::dict d;
            d["xy_extents"] = std::make_tuple(cc.xy_extents[0], cc.xy_extents[1]);
            d["mesh_type"] = meshTypeName(cc.mesh_type);
            d["mesh_id"] = cc.mesh_id;
            std::vector<int> mat_ids;
            for (auto const id : cc.material_ids) {
              mat_ids.push_back(static_cast<int>(id));
            }
            d["material_ids"] = mat_ids;
            return d;
          },
          py::arg("index"));

  //============================================================================
  // gmsh
  //============================================================================

  py::module_ gm = m.def_submodule("gmsh", "UM2 operations on the current gmsh model");

  gm.def(
      "overlay_coarse_grid",
      [](um2::mpact::Model const & model, um2::Material const & fill) {
        py::gil_scoped_release nogil;
        ErrorScope scope;
        um2::gmsh::model::occ::overlayCoarseGrid(model, fill);
        ErrorScope::check();
      },
      py::arg("model"), py::arg("fill_material"),
      "Overlay the coarse grid of an MPACT model on the gmsh model, labeling "
      "each coarse cell and filling empty space with fill_material.");

  gm.def(
      "set_global_mesh_size",
      [](double size) {
        py::gil_scoped_release nogil;
        ErrorScope scope;
        um2::gmsh::model::mesh::setGlobalMeshSize(size);
        ErrorScope::check();
      },
      py::arg("size"));

  gm.def(
      "set_mesh_field_from_knudsen_number",
      [](int dim, std::vector<um2::Material> const & materials, double kn_target,
         double fuel_mfp_threshold, double fuel_mfp_scale, double abs_mfp_threshold,
         double abs_mfp_scale) {
        py::gil_scoped_release nogil;
        ErrorScope scope;
        int const field = um2::gmsh::model::mesh::setMeshFieldFromKnudsenNumber(
            dim, toVector(materials), kn_target, fuel_mfp_threshold, fuel_mfp_scale,
            abs_mfp_threshold, abs_mfp_scale);
        ErrorScope::check();
        return field;
      },
      py::arg("dim"), py::arg("materials"), py::arg("kn_target"),
      py::arg("fuel_mfp_threshold") = -1.0, py::arg("fuel_mfp_scale") = -1.0,
      py::arg("abs_mfp_threshold") = -1.0, py::arg("abs_mfp_scale") = -1.0);

  gm.def(
      "generate_mesh",
      [](um2::MeshType mesh_type, int smooth_iters) {
        py::gil_scoped_release nogil;
        ErrorScope scope;
        um2::gmsh::model::mesh::generateMesh(mesh_type, smooth_iters);
        ErrorScope::check();
      },
      py::arg("mesh_type"), py::arg("smooth_iters") = 100);

  gm.def(
      "write",
      [](std::string const & filename, bool extra_info) {
        py::gil_scoped_release nogil;
        ErrorScope scope;
        um2::gmsh::write(filename, extra_info);
        ErrorScope::check();
      },
      py::arg("filename"), py::arg("extra_info") = false);

  gm.def(
      "get_materials",
      []() {
        std::vector<um2::Material> materials;
        py::gil_scoped_release nogil;
        ErrorScope scope;
        um2::gmsh::model::getMaterials(materials);
        ErrorScope::check();
        return materials;
      },
      "Materials (names and colors) of the Material_ physical groups of the gmsh "
      "model.");

  gm.def(
      "count_entities",
      [](int dim) {
        um2::gmsh::vectorpair dimtags;
        um2::gmsh::model::getEntities(dimtags, dim);
        return dimtags.size();
      },
      py::arg("dim") = -1,
      "Number of entities in the current gmsh model, as seen by UM2. Used to check "
      "that UM2 and the gmsh Python API share the same model.");
}
