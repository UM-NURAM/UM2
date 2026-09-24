#include <um2/physics/cross_section.hpp>
#include <um2/physics/mpact_user_xs.hpp>

#include <fstream>

#include "../test_macros.hpp"

Float constexpr eps = 1e-6;

TEST_CASE(read_c5g7)
{
  um2::mpact::UserXSLibrary const lib("./xs_files/c5g7.xsl");
  ASSERT(lib.title() == um2::String("C5G7 macroscopic cross section data"));
  ASSERT(lib.numGroups() == 7);
  ASSERT(lib.numSets() == 8);
  ASSERT_NEAR(lib.groupBounds()[0], 2.0e7, eps);
  ASSERT_NEAR(lib.groupBounds()[6], 0.0635, eps);
  ASSERT(lib.names()[0] == um2::String("UO2-3.3"));
  ASSERT(lib.names()[7] == um2::String("CRod"));
  ASSERT(lib.getIndex("Moderator") == 6);
  ASSERT(lib.getIndex("NotASet") == -1);
  for (auto const order : lib.scatteringOrders()) {
    ASSERT(order == 0);
  }

  // Compare against the C5G7 data built into UM2. The .xsl file has the
  // same data, except for the order of the sets and the (unused) control rod.
  auto const c5g7 = um2::getC5G7XSecs();
  um2::Vector<um2::String> const names = {"UO2-3.3",  "MOX-4.3",   "MOX-7.0",  "MOX-8.7",
                                          "FissCham", "GuideTube", "Moderator"};
  for (Int i = 0; i < names.size(); ++i) {
    auto const & ref = c5g7[i];
    auto const & xs = lib.getXSec(names[i]);
    ASSERT(xs.isMacro());
    ASSERT(xs.isFissile() == ref.isFissile());
    for (Int g = 0; g < 7; ++g) {
      ASSERT_NEAR(xs.a()[g], ref.a()[g], eps);
      ASSERT_NEAR(xs.f()[g], ref.f()[g], eps);
      // nu-fission in UM2 is nu * fission, so only agrees to ~6 digits
      ASSERT_NEAR(xs.nuf()[g], ref.nuf()[g], 1e-5 * ref.nuf()[g] + eps);
      ASSERT_NEAR(xs.s()[g], ref.s()[g], eps);
      // total = absorption + out-scattering, which matches the benchmark
      // transport cross section to the precision of the tabulated data
      ASSERT_NEAR(xs.tr()[g], ref.tr()[g], 1e-4 * ref.tr()[g]);
      for (Int gp = 0; gp < 7; ++gp) {
        ASSERT_NEAR(xs.ss()(g, gp), ref.ss()(g, gp), eps);
      }
    }
  }
  // Fission spectrum
  ASSERT_NEAR(lib.chis()[0][0], 5.8791e-1, eps);
  ASSERT_NEAR(lib.chis()[0][3], 1.1761e-7, eps);
  ASSERT_NEAR(lib.chis()[5][0], 0.0, eps);
}

TEST_CASE(read_p1_with_comments)
{
  // A small 2-group library with a P1 set, inline comments, and wrapped lines
  {
    std::ofstream file("p1_test.xsl");
    file << "Two group test ! not a comment in the title\n"
         << " 2 2\n"
         << " 2.0E+07 0.625\n"
         << "! comment line\n"
         << "XSMACRO fuel 1 ! trailing comment\n"
         << " 0.01 0.02 0.008 1.0\n"
         << " 0.10 0.20 0.080 0.0\n"
         << " 0.30 0.00\n"
         << " 0.02 0.40\n"
         << " 0.05 0.00 0.01\n" // P1 matrix, wrapped differently
         << " 0.03\n"
         << "XSMACRO water 0\n"
         << " 0.001 0.0 0.0 0.0\n"
         << " 0.020 0.0 0.0 0.0\n"
         << " 0.50 0.00\n"
         << " 0.10 1.00\n";
  }
  um2::mpact::UserXSLibrary const lib("p1_test.xsl");
  ASSERT(lib.title() == um2::String("Two group test ! not a comment in the title"));
  ASSERT(lib.numSets() == 2);
  ASSERT(lib.scatteringOrders()[0] == 1);
  ASSERT(lib.scatteringOrders()[1] == 0);
  auto const & fuel = lib.getXSec("fuel");
  ASSERT(fuel.isFissile());
  ASSERT_NEAR(fuel.ss()(1, 0), 0.02, eps); // 1 -> 2
  ASSERT_NEAR(fuel.ss()(0, 1), 0.00, eps); // 2 -> 1
  ASSERT_NEAR(fuel.s()[0], 0.32, eps);
  ASSERT_NEAR(fuel.tr()[0], 0.33, eps);
  ASSERT_NEAR(fuel.tr()[1], 0.50, eps);
  auto const & water = lib.getXSec("water");
  ASSERT(!water.isFissile());
  ASSERT_NEAR(water.ss()(0, 1), 0.00, eps);
  ASSERT_NEAR(water.ss()(1, 0), 0.10, eps);
  ASSERT_NEAR(water.tr()[0], 0.601, eps);
}

TEST_SUITE(mpact_user_xs)
{
  TEST(read_c5g7);
  TEST(read_p1_with_comments);
}

auto
main() -> int
{
  RUN_SUITE(mpact_user_xs);
  return 0;
}
