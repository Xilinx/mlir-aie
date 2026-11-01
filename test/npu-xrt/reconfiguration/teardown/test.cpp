// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs aie.mlir's configurations and checks each one's slice of the output.
//
//   ./test sequence  runs main:sequence three times.
//   ./test init      runs main:reconfig_init once, then as `sequence`.
//   ./test switch E  as `init`, but before each later pass runs E's
//                    main:sequence in another hardware context.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int NUM_CONFIGS = 3;
constexpr int SLICE = 16;
constexpr int COUNT = NUM_CONFIGS * SLICE;
constexpr int32_t POISON = -1;

// The memtile's access pattern of each configuration, outermost dimension
// first, as (size, stride) pairs.
static const std::pair<int, int> PATTERNS[NUM_CONFIGS][2] = {
    {{4, 4}, {4, 1}}, {{4, 1}, {4, 4}}, {{8, 1}, {2, 8}}};

// Counts the elements of configuration `config`'s slice that are wrong.
static int check(const int32_t *in, const int32_t *out, int config) {
  auto [outer, inner] = PATTERNS[config - 1];
  const int32_t *slice_in = in + (config - 1) * SLICE;
  const int32_t *slice_out = out + (config - 1) * SLICE;
  int wrong = 0;
  for (int a = 0; a < outer.first; a++)
    for (int b = 0; b < inner.first; b++) {
      int m = a * inner.first + b;
      wrong +=
          slice_out[m] != slice_in[a * outer.second + b * inner.second] + 100;
    }
  return wrong;
}

int main(int argc, const char *argv[]) {
  std::string mode = argc > 1 ? argv[1] : "sequence";
  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context context(device, elf);
  xrt::bo bo_in = xrt::ext::bo{device, COUNT * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, COUNT * sizeof(int32_t)};
  int32_t *in = bo_in.map<int32_t *>();
  int32_t *out = bo_out.map<int32_t *>();
  for (int i = 0; i < COUNT; i++)
    in[i] = 1000 * (i / SLICE) + i % SLICE;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::optional<xrt::hw_context> other;
  if (mode == "switch" && argc > 2)
    other.emplace(device, xrt::elf{argv[2]});

  auto dispatch = [&](xrt::hw_context &on, const std::string &name, bool args) {
    std::fill(out, out + COUNT, POISON);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto kernel = xrt::ext::kernel(on, name);
    xrt::run run(kernel);
    if (args) {
      run.set_arg(0, bo_in);
      run.set_arg(1, bo_out);
    }
    run.start();
    run.wait2();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  };
  int failures = 0;
  auto report = [&](const std::string &what, int config) {
    if (int wrong = check(in, out, config)) {
      std::cout << what << ", configuration " << config << ": " << wrong
                << " wrong" << std::endl;
      failures++;
    }
  };

  if (mode != "sequence" && mode != "init" && !other) {
    std::cerr << "unknown mode " << mode << std::endl;
    return 2;
  }
  if (mode != "sequence")
    dispatch(context, "main:reconfig_init", false);
  for (int pass = 0; pass < 3; pass++) {
    if (other && pass > 0)
      dispatch(*other, "main:sequence", true);
    dispatch(context, "main:sequence", true);
    for (int i = 1; i <= NUM_CONFIGS; i++)
      report("pass " + std::to_string(pass), i);
  }

  std::cout << (failures ? "FAIL!" : "PASS!") << std::endl;
  return failures ? 1 : 0;
}
