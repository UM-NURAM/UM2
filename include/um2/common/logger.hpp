#pragma once

#include <um2/common/settings.hpp>

#include <chrono>

//==============================================================================
// LOG
//==============================================================================
// A simple logger for use in host code.
// The logger can be configured to:
//  - log messages of different verbosity levels
//  - prefix messages with a timestamp
//  - colorize messages based on their verbosity level
//  - exit the program after an error is logged (or not)
//  - record the first error message since clearErrors(), so that callers which disable
//    exit_on_error (e.g. the Python bindings) can detect and report errors
//
// The logger:
// - can be configured at compile time by defining the MAX_LOG_LEVEL macro.
// - is not thread-safe.
// - is just a fixed-size buffer that is filled with the message and then
//    printed. The message arguments are converted using the toBuffer function.

namespace um2::logger
{

using Clock = std::chrono::system_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

namespace levels
{
inline constexpr int32_t off = 0;   // no messages
inline constexpr int32_t error = 1; // only errors
inline constexpr int32_t warn = 2;  // errors and warnings
inline constexpr int32_t info = 3;  // errors, warnings and info
inline constexpr int32_t debug = 4; // errors, warnings, info and debug
} // namespace levels

//==============================================================================
// Global variables
//==============================================================================

// Suppress warnings for non-const global variables, since this is a global logger
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

extern int32_t & level;
extern bool & timestamped;
extern bool & colorized;
extern bool & exit_on_error;
extern TimePoint start_time;

inline constexpr int32_t buffer_size = 256;
extern char buffer[buffer_size];
extern char const * const buffer_end; // 1 past the last valid character in the buffer
extern int32_t error_count;            // Number of errors logged since clearErrors()
extern char first_error[buffer_size];  // First error since clearErrors() (no preamble)

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// Functions
//==============================================================================

PURE constexpr auto
getLastMessage() noexcept -> StringView
{
  return {buffer};
}

// Reset the logger to its default state
void
reset() noexcept;

// Reset error_count and first_error
void
clearErrors() noexcept;

// Increment error_count and, if it is the first error, store the message in
// [first, last)
void
recordError(char const * first, char const * last) noexcept;

// Write types to the buffer
template <class T>
auto
toBuffer(char * buffer_pos, T const & value) noexcept -> char *;

// Handle fixed-size character arrays by treating them as pointer
template <uint64_t N>
auto
toBuffer(char * buffer_pos, char const (&value)[N]) noexcept -> char *
{
  char const * const p = value;
  return toBuffer(buffer_pos, p);
}

// Add the timestamp to the buffer if the log is timestamped
auto
addTimestamp(char * buffer_pos) noexcept -> char *;

// Add color to the buffer if the log is colorized
auto
addColor(int32_t msg_level, char * buffer_pos) noexcept -> char *;

// Add the log level to the buffer
auto
addLevel(int32_t msg_level, char * buffer_pos) noexcept -> char *;

// Set the preamble of the message
auto
setPreamble(int32_t msg_level) noexcept -> char *;

// Set the postamble of the message
void
setPostamble(char * buffer_pos) noexcept;

// Print the message
template <class... Args>
void
printMessage(int32_t const msg_level, Args const &... args) noexcept
{
  // Errors are always recorded, even if they are not printed.
  if (msg_level <= level || msg_level == levels::error) {
    char * buffer_pos = setPreamble(msg_level);
    char const * const msg_start = buffer_pos;

    // Use fold expression to send each argument to the buffer.
    // We need a lambda function to capture the buffer_pos variable, since it is
    // not a template parameter.
    ([&buffer_pos](auto const & arg) { buffer_pos = toBuffer(buffer_pos, arg); }(args),
     ...);

    if (msg_level == levels::error) {
      recordError(msg_start, buffer_pos);
    }

    setPostamble(buffer_pos);

    if (msg_level > level) {
      return;
    }

    // Print the message
    int fprintf_result = 0;
    if (msg_level == levels::error) {
      fprintf_result = fprintf(stderr, "%s\n", buffer);
      if (exit_on_error) {
        exit(1);
      }
    } else {
      fprintf_result = fprintf(stdout, "%s\n", buffer);
    }
#if UM2_ENABLE_ASSERTS
    ASSERT(fprintf_result > 0);
#else
    if (fprintf_result == 0) {
      exit(1);
    }
#endif
  } // msg_level <= level
} // printMessage

template <class... Args>
void
error(Args const &... args) noexcept
{
  printMessage(levels::error, args...);
}

template <class... Args>
void
warn(Args const &... args) noexcept
{
  printMessage(levels::warn, args...);
}

template <class... Args>
void
info(Args const &... args) noexcept
{
  printMessage(levels::info, args...);
}

template <class... Args>
void
debug(Args const &... args) noexcept
{
  printMessage(levels::debug, args...);
}

} // namespace um2::logger

#if MAX_LOG_LEVEL > 0
#  define LOG_ERROR(...) um2::logger::error(__VA_ARGS__)
#else
#  define LOG_ERROR(...)
#endif

#if MAX_LOG_LEVEL > 1
#  define LOG_WARN(...) um2::logger::warn(__VA_ARGS__)
#else
#  define LOG_WARN(...)
#endif

#if MAX_LOG_LEVEL > 2
#  define LOG_INFO(...) um2::logger::info(__VA_ARGS__)
#else
#  define LOG_INFO(...)
#endif

#if MAX_LOG_LEVEL > 3
#  define LOG_DEBUG(...) um2::logger::debug(__VA_ARGS__)
#else
#  define LOG_DEBUG(...)
#endif
