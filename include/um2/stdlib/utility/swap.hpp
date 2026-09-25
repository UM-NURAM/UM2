#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/utility/move.hpp>

#include <cstdlib>     // std::size_t
#include <type_traits> // std::is_trivially_move_constructible_v,
                       // std::is_trivially_move_assignable_v

namespace um2
{

// The swap functions are declared in a nested namespace and made available as
// um2::swap by a using-directive. This way, they are found by qualified calls
// (um2::swap(a, b)), but not by argument-dependent lookup for um2 types. With
// argument-dependent lookup, unqualified calls such as those in std::sort
// (using std::swap; swap(a, b)) are ambiguous with some standard libraries
// (e.g. GCC 16).
namespace swap_impl
{

template <class T>
  requires(std::is_trivially_move_constructible_v<T> &&
           std::is_trivially_move_assignable_v<T>)
HOSTDEV constexpr void
swap(T & a, T & b) noexcept
{
  T tmp = um2::move(a);
  a = um2::move(b);
  b = um2::move(tmp);
}

template <class T, size_t N>
HOSTDEV constexpr void
swap(T (&a)[N], T (&b)[N]) noexcept
{
  for (size_t i = 0; i < N; ++i) {
    swap_impl::swap(a[i], b[i]);
  }
}

} // namespace swap_impl

// NOLINTNEXTLINE(google-build-using-namespace) See above
using namespace swap_impl;

} // namespace um2
