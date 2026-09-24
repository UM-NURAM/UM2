#pragma once

#include <um2/physics/cross_section.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

//======================================================================
// MPACT USER CROSS SECTION LIBRARY
//======================================================================
// A macroscopic multi-group cross section library in the MPACT "user"
// format (.xsl). The format is:
//
//  Line 1: Title
//  Line 2: num_groups num_sets
//  Line 3: Upper energy bound of each group [eV], in descending order. The
//          lower bound of the lowest energy group is assumed to be 0.
//  Then, for each cross section set:
//    XSMACRO <name> <scattering order N>
//    num_groups lines of: absorption nu-fission fission chi
//    (N + 1) scattering matrices (P0, P1, ..., PN), each of which is
//    num_groups lines of num_groups values. Row i, column j is the
//    cross section for scattering from group j to group i.
//
// Comments start with '!' and continue to the end of the line.
// The total cross section is defined as absorption + P0 out-scattering.

namespace um2::mpact
{

class UserXSLibrary
{

  String _title;
  Vector<Float> _group_bounds;    // Upper energy bounds [eV]. size = numGroups()
  Vector<String> _names;          // Name of each cross section set
  Vector<XSec> _xsecs;            // Macroscopic cross sections (P0 scattering)
  Vector<Vector<Float>> _chis;    // Fission spectrum of each set
  Vector<Int> _scattering_orders; // Scattering order of each set

public:
  //======================================================================
  // Constructors
  //======================================================================

  UserXSLibrary() noexcept = default;

  explicit UserXSLibrary(String const & filename);

  //======================================================================
  // Accessors
  //======================================================================

  PURE [[nodiscard]] auto
  title() const noexcept -> String const &
  {
    return _title;
  }

  PURE [[nodiscard]] auto
  groupBounds() const noexcept -> Vector<Float> const &
  {
    return _group_bounds;
  }

  PURE [[nodiscard]] auto
  numGroups() const noexcept -> Int
  {
    return _group_bounds.size();
  }

  PURE [[nodiscard]] auto
  numSets() const noexcept -> Int
  {
    return _names.size();
  }

  PURE [[nodiscard]] auto
  names() const noexcept -> Vector<String> const &
  {
    return _names;
  }

  PURE [[nodiscard]] auto
  xsecs() const noexcept -> Vector<XSec> const &
  {
    return _xsecs;
  }

  PURE [[nodiscard]] auto
  chis() const noexcept -> Vector<Vector<Float>> const &
  {
    return _chis;
  }

  PURE [[nodiscard]] auto
  scatteringOrders() const noexcept -> Vector<Int> const &
  {
    return _scattering_orders;
  }

  //======================================================================
  // Methods
  //======================================================================

  // Read the library from a file. Any existing data is discarded.
  void
  read(String const & filename);

  // Get the index of the set with the given name. Returns -1 if not found.
  PURE [[nodiscard]] auto
  getIndex(String const & name) const noexcept -> Int;

  // Get the cross section of the set with the given name.
  PURE [[nodiscard]] auto
  getXSec(String const & name) const noexcept -> XSec const &;
};

} // namespace um2::mpact
