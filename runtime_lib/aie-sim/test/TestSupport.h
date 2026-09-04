//===- TestSupport.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Assertions for the aie-sim tests. Deliberately tiny: these are plain
// executables run by CTest, matching test/CppTests, and pulling in a test
// framework for them would be a new dependency for no gain.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_TEST_TESTSUPPORT_H
#define AIESIM_TEST_TESTSUPPORT_H

#include <cstdio>
#include <cstdlib>
#include <string>

namespace aiesim_test {

inline int &failureCount() {
  static int failures = 0;
  return failures;
}

inline void reportFailure(const char *file, int line, const std::string &what) {
  std::fprintf(stderr, "%s:%d: FAIL: %s\n", file, line, what.c_str());
  ++failureCount();
}

inline int summarize(const char *name) {
  if (failureCount() == 0) {
    std::printf("%s: PASS\n", name);
    return 0;
  }
  std::printf("%s: %d failure(s)\n", name, failureCount());
  return 1;
}

} // namespace aiesim_test

#define AIESIM_CHECK(cond)                                                     \
  do {                                                                         \
    if (!(cond))                                                               \
      ::aiesim_test::reportFailure(__FILE__, __LINE__, #cond);                 \
  } while (0)

#define AIESIM_CHECK_EQ(a, b)                                                  \
  do {                                                                         \
    auto lhsValue = (a);                                                       \
    auto rhsValue = (b);                                                       \
    if (!(lhsValue == rhsValue))                                               \
      ::aiesim_test::reportFailure(__FILE__, __LINE__,                         \
                                   std::string(#a) + " == " + #b + " (got " +  \
                                       std::to_string(lhsValue) + " vs " +     \
                                       std::to_string(rhsValue) + ")");        \
  } while (0)

#endif // AIESIM_TEST_TESTSUPPORT_H
