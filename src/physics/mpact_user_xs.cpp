#include <um2/common/logger.hpp>
#include <um2/common/strto.hpp>
#include <um2/config.hpp>
#include <um2/physics/cross_section.hpp>
#include <um2/physics/mpact_user_xs.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/vector.hpp>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

namespace um2::mpact
{

namespace
{

// Remove a '!' comment from the line
void
stripComment(std::string & line)
{
  auto const pos = line.find('!');
  if (pos != std::string::npos) {
    line.erase(pos);
  }
}

// Reads whitespace separated tokens from the file, skipping comments.
class TokenStream
{
  std::ifstream & _file;
  std::istringstream _line;
  Int _line_number = 0;

public:
  explicit TokenStream(std::ifstream & file)
      : _file(file)
  {
  }

  // Get the next token. Returns false if the end of the file is reached.
  auto
  next(std::string & token) -> bool
  {
    while (!(_line >> token)) {
      std::string line;
      if (!std::getline(_file, line)) {
        return false;
      }
      ++_line_number;
      stripComment(line);
      _line.clear();
      _line.str(line);
    }
    return true;
  }

  // Get the next token as a number. Logs an error and returns false on failure.
  template <class T>
  auto
  nextNumber(T & value, String const & filename) -> bool
  {
    std::string token;
    if (!next(token)) {
      LOG_ERROR("Unexpected end of file in ", filename);
      return false;
    }
    char * end = nullptr;
    value = strto<T>(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0') {
      LOG_ERROR("Expected a number but found '", token.c_str(), "' on line ",
                _line_number, " of ", filename);
      return false;
    }
    return true;
  }

  [[nodiscard]] auto
  lineNumber() const noexcept -> Int
  {
    return _line_number;
  }

  // Set the number of lines already read from the file
  void
  setLineNumber(Int n) noexcept
  {
    _line_number = n;
  }
};

} // namespace

UserXSLibrary::UserXSLibrary(String const & filename) { read(filename); }

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
UserXSLibrary::read(String const & filename)
{
  LOG_INFO("Reading MPACT user cross section library: ", filename);

  _title = String();
  _group_bounds.clear();
  _names.clear();
  _xsecs.clear();
  _chis.clear();
  _scattering_orders.clear();

  std::ifstream file(filename.data());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: ", filename);
    return;
  }

  // Line 1: title. Comments are not stripped from the title.
  std::string line;
  if (!std::getline(file, line)) {
    LOG_ERROR("Empty MPACT user cross section library: ", filename);
    return;
  }
  // Remove leading/trailing whitespace
  auto const first = line.find_first_not_of(" \t\r");
  auto const last = line.find_last_not_of(" \t\r");
  if (first != std::string::npos) {
    _title = String(line.substr(first, last - first + 1).c_str());
  }

  TokenStream tokens(file);
  tokens.setLineNumber(1);

  // Line 2: num_groups num_sets
  Int num_groups = 0;
  Int num_sets = 0;
  if (!tokens.nextNumber(num_groups, filename) || !tokens.nextNumber(num_sets, filename)) {
    return;
  }
  if (num_groups <= 0 || num_sets < 0) {
    LOG_ERROR("Invalid number of groups (", num_groups, ") or cross section sets (",
              num_sets, ") in ", filename);
    return;
  }

  // Line 3: group upper bounds
  _group_bounds.resize(num_groups);
  for (Int ig = 0; ig < num_groups; ++ig) {
    if (!tokens.nextNumber(_group_bounds[ig], filename)) {
      return;
    }
    if (ig > 0 && !(_group_bounds[ig] < _group_bounds[ig - 1])) {
      LOG_ERROR("Group bounds are not strictly decreasing in ", filename);
      return;
    }
  }
  if (!(_group_bounds[num_groups - 1] > 0)) {
    LOG_ERROR("Group bounds must be positive in ", filename);
    return;
  }

  // Cross section sets
  std::string token;
  while (tokens.next(token)) {
    if (token == "XSMICRO") {
      LOG_ERROR("XSMICRO blocks are not supported (line ", tokens.lineNumber(), " of ",
                filename, ")");
      return;
    }
    if (token != "XSMACRO") {
      LOG_ERROR("Expected XSMACRO but found '", token.c_str(), "' on line ",
                tokens.lineNumber(), " of ", filename);
      return;
    }
    std::string name;
    if (!tokens.next(name)) {
      LOG_ERROR("Unexpected end of file in ", filename);
      return;
    }
    Int order = 0;
    if (!tokens.nextNumber(order, filename)) {
      return;
    }
    if (order < 0) {
      LOG_ERROR("Negative scattering order for set ", name.c_str(), " in ", filename);
      return;
    }

    XSec xsec(num_groups);
    xsec.isMacro() = true;
    Vector<Float> chi(num_groups, 0);
    // absorption nu-fission fission chi
    for (Int ig = 0; ig < num_groups; ++ig) {
      if (!tokens.nextNumber(xsec.a()[ig], filename) ||
          !tokens.nextNumber(xsec.nuf()[ig], filename) ||
          !tokens.nextNumber(xsec.f()[ig], filename) ||
          !tokens.nextNumber(chi[ig], filename)) {
        return;
      }
      if (xsec.f()[ig] > 0 || xsec.nuf()[ig] > 0) {
        xsec.isFissile() = true;
      }
    }
    // Scattering matrices. Only P0 is stored; higher orders are read and
    // discarded, since UM2 does not use them.
    for (Int l = 0; l <= order; ++l) {
      for (Int ig_to = 0; ig_to < num_groups; ++ig_to) {
        for (Int ig_from = 0; ig_from < num_groups; ++ig_from) {
          Float value = 0;
          if (!tokens.nextNumber(value, filename)) {
            return;
          }
          if (l == 0) {
            xsec.ss()(ig_to, ig_from) = value;
          }
        }
      }
    }
    // Total scattering is the sum of each column. Total = absorption + scattering.
    for (Int ig_from = 0; ig_from < num_groups; ++ig_from) {
      for (Int ig_to = 0; ig_to < num_groups; ++ig_to) {
        xsec.s()[ig_from] += xsec.ss()(ig_to, ig_from);
      }
      xsec.tr()[ig_from] = xsec.a()[ig_from] + xsec.s()[ig_from];
    }
    xsec.validate();

    _names.emplace_back(name.c_str());
    _xsecs.push_back(um2::move(xsec));
    _chis.push_back(um2::move(chi));
    _scattering_orders.push_back(order);
  }

  if (_names.size() != num_sets) {
    LOG_ERROR("Expected ", num_sets, " cross section sets but found ", _names.size(),
              " in ", filename);
    return;
  }
  LOG_INFO("Read ", num_sets, " cross section sets with ", num_groups, " groups");
}

PURE [[nodiscard]] auto
UserXSLibrary::getIndex(String const & name) const noexcept -> Int
{
  for (Int i = 0; i < _names.size(); ++i) {
    if (_names[i] == name) {
      return i;
    }
  }
  return -1;
}

PURE [[nodiscard]] auto
UserXSLibrary::getXSec(String const & name) const noexcept -> XSec const &
{
  Int const i = getIndex(name);
  if (i < 0) {
    LOG_ERROR("Cross section set not found: ", name);
    static XSec const empty{};
    return empty;
  }
  return _xsecs[i];
}

} // namespace um2::mpact
