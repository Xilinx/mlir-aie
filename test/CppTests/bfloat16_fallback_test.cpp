//===- bfloat16_fallback_test.cpp -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Exercises the uint16-storage fallback path of test_utils::bfloat16_t.
// Built with -U__STDCPP_BFLOAT16_T__ so the fallback branch is selected even
// on toolchains that provide std::bfloat16_t natively (i.e. Linux CI). MSVC
// reaches this branch unconditionally.

#include "test_utils.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

struct RoundTripCase {
  float value;
  std::uint16_t expected_bits;
  const char *label;
};

int check_from_float(const RoundTripCase &c) {
  const test_utils::bfloat16_t got = test_utils::bfloat16_from_float(c.value);
  if (static_cast<std::uint16_t>(got) != c.expected_bits) {
    std::printf("FAIL from_float(%s=%g): got 0x%04x, expected 0x%04x\n",
                c.label, c.value, static_cast<unsigned>(got),
                static_cast<unsigned>(c.expected_bits));
    return 1;
  }
  return 0;
}

int check_to_float_finite(std::uint16_t bits, float expected,
                          const char *label) {
  const test_utils::bfloat16_t b = test_utils::bfloat16_from_bits(bits);
  const float got = test_utils::bfloat16_to_float(b);
  if (got != expected) {
    std::printf("FAIL to_float(%s=0x%04x): got %g, expected %g\n", label,
                static_cast<unsigned>(bits), got, expected);
    return 1;
  }
  return 0;
}

int check_rtne_tie(std::uint32_t f32_bits, std::uint16_t expected_bits,
                   const char *label) {
  float value;
  std::memcpy(&value, &f32_bits, sizeof(value));
  const test_utils::bfloat16_t got = test_utils::bfloat16_from_float(value);
  if (static_cast<std::uint16_t>(got) != expected_bits) {
    std::printf(
        "FAIL rtne(%s, f32=0x%08x): got 0x%04x, expected 0x%04x (even)\n",
        label, f32_bits, static_cast<unsigned>(got),
        static_cast<unsigned>(expected_bits));
    return 1;
  }
  return 0;
}

} // namespace

int main() {
  int failures = 0;

  const RoundTripCase finite_cases[] = {
      {0.0f, 0x0000, "+0"},    {-0.0f, 0x8000, "-0"}, {1.0f, 0x3F80, "1.0"},
      {-1.0f, 0xBF80, "-1.0"}, {0.5f, 0x3F00, "0.5"}, {2.0f, 0x4000, "2.0"},
      {-2.0f, 0xC000, "-2.0"},
  };
  for (const auto &c : finite_cases) {
    failures += check_from_float(c);
    failures += check_to_float_finite(c.expected_bits, c.value, c.label);
  }

  // RTNE ties: low 16 bits of the f32 representation are exactly 0x8000, so
  // the result must round to the even bf16 mantissa LSB.
  // 0x3F808000 sits halfway between 0x3F80 (1.0) and 0x3F81; even is 0x3F80.
  failures += check_rtne_tie(0x3F808000u, 0x3F80u, "1.0 + 1 ulp tie -> even");
  // 0x3F818000 sits halfway between 0x3F81 and 0x3F82; even is 0x3F82.
  failures += check_rtne_tie(0x3F818000u, 0x3F82u, "1.0 + 3 ulp tie -> even");

  // +Inf / -Inf preserved.
  {
    const float pos_inf = std::numeric_limits<float>::infinity();
    const test_utils::bfloat16_t b = test_utils::bfloat16_from_float(pos_inf);
    if (static_cast<std::uint16_t>(b) != 0x7F80u) {
      std::printf("FAIL +Inf: got 0x%04x, expected 0x7F80\n",
                  static_cast<unsigned>(b));
      failures++;
    }
    const test_utils::bfloat16_t bn = test_utils::bfloat16_from_float(-pos_inf);
    if (static_cast<std::uint16_t>(bn) != 0xFF80u) {
      std::printf("FAIL -Inf: got 0x%04x, expected 0xFF80\n",
                  static_cast<unsigned>(bn));
      failures++;
    }
  }

  // NaN: exponent stays all-ones, mantissa stays nonzero. Exact bits depend on
  // input mantissa, so just check the structural invariant.
  {
    const float nan = std::nanf("");
    const test_utils::bfloat16_t b = test_utils::bfloat16_from_float(nan);
    const auto bits = static_cast<std::uint16_t>(b);
    const bool exp_all_ones = ((bits >> 7) & 0xFFu) == 0xFFu;
    const bool mantissa_nonzero = (bits & 0x7Fu) != 0u;
    if (!exp_all_ones || !mantissa_nonzero) {
      std::printf("FAIL NaN: got 0x%04x (expected exp=0xFF, mantissa!=0)\n",
                  static_cast<unsigned>(bits));
      failures++;
    }
  }

  if (failures > 0) {
    std::printf("bfloat16 fallback: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("bfloat16 fallback: all checks passed\n");
  return 0;
}
